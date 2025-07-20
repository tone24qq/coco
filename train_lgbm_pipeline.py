#!/usr/bin/env python
"""Enhanced LightGBM pipeline for board completion with improved features and training strategy."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class Float32StandardScaler:
    """Lightweight scaler that keeps data in float32 precision."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Float32StandardScaler":
        X = X.astype(np.float32, copy=False)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted")
        X = X.astype(np.float32, copy=False)
        X -= self.mean_
        X /= self.scale_
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


# Improved hyperparameters
NEG_RATIO = 2  # Reduced from 3 to 2
SHARD_SIZE = 50_000  # Reduced for better memory management
TREES_PER = 500  # Increased from 200

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")


def _yield_json(fp) -> Iterator[object]:
    """Parse JSON data from file pointer."""
    raw = fp.read()
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            yield from arr
            return
    except Exception:
        pass
    try:
        text = raw.decode()
    except AttributeError:
        text = raw
    try:
        rows = [
            [int(x) for x in ln.split()]
            for ln in text.strip().splitlines()
            if ln.strip()
        ]
        if rows and all(len(r) == len(rows[0]) for r in rows):
            yield rows
            return
    except Exception:
        pass
    try:
        fp.seek(0)
    except Exception:
        return
    for ln in fp:
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def iter_objects(root: Path) -> Iterator[Tuple[np.ndarray, int | None]]:
    """Iterate over board objects from files."""

    def _parse(obj: object) -> Iterator[Tuple[np.ndarray, int | None]]:
        if isinstance(obj, dict) and "board" in obj:
            bd = np.asarray(obj["board"], dtype=int)
            tgt = obj.get("target")
            yield bd, tgt
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], list) and isinstance(obj[0][0], list):
                for board in obj:
                    yield np.asarray(board, dtype=int), None
            elif obj and isinstance(obj[0], list):
                yield np.asarray(obj, dtype=int), None
            elif obj and isinstance(obj[0], (int, float)):
                arr = np.asarray(obj, dtype=int)
                n = arr.size
                r = int(math.isqrt(n))
                while r > 1 and n % r:
                    r -= 1
                yield arr.reshape(r, n // r), None

    for p in root.rglob("*.zip"):
        with zipfile.ZipFile(p) as zf:
            for nm in zf.namelist():
                with zf.open(nm) as fp:
                    for item in _yield_json(fp):
                        yield from _parse(item)
    for p in root.rglob("*.json"):
        with p.open("rb") as fp:
            for item in _yield_json(fp):
                yield from _parse(item)


def _get_all_candidates(masked: np.ndarray, pos: Tuple[int, int]) -> Set[int]:
    """Get all possible candidates for a position based on constraints."""
    r, c = pos
    R, C = masked.shape
    max_val = R * C

    all_nums = set(range(1, max_val + 1))
    row_used = set(masked[r, :][masked[r, :] != -1])
    col_used = set(masked[:, c][masked[:, c] != -1])
    # For general boards, use intersection of constraints
    candidates = all_nums - row_used - col_used
    # Remove globally over-used numbers
    return candidates


def _get_8_neighbors(masked: np.ndarray, pos: Tuple[int, int]) -> List[int]:
    """Return non-masked values from the 8-neighborhood."""
    r, c = pos
    R, C = masked.shape
    neighbors = []
    directions = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C and masked[nr, nc] != -1:
            neighbors.append(int(masked[nr, nc]))
    return neighbors


def _eight_neighbor_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Features derived from 8 surrounding cells."""
    neighbors = _get_8_neighbors(masked, pos)

    if not neighbors:
        return [0.0] * 10

    arr = np.asarray(neighbors, dtype=float)

    neighbor_mean = float(np.mean(arr))
    neighbor_std = float(np.std(arr)) if len(arr) > 1 else 0.0

    diff = np.abs(arr - target)
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff)) if len(arr) > 1 else 0.0

    odd_count = np.sum(arr % 2 == 1)
    even_count = len(arr) - odd_count
    odd_ratio = float(odd_count / len(arr))
    even_ratio = float(even_count / len(arr))

    continuity_score = 0.0
    if len(arr) >= 2:
        sorted_neighbors = np.sort(arr)
        consecutive_pairs = 0
        for i in range(len(sorted_neighbors) - 1):
            if sorted_neighbors[i + 1] - sorted_neighbors[i] == 1:
                consecutive_pairs += 1
        continuity_score = float(consecutive_pairs / (len(sorted_neighbors) - 1))

    min_neighbor = float(np.min(arr))
    max_neighbor = float(np.max(arr))
    target_to_min_diff = abs(target - min_neighbor)
    target_to_max_diff = abs(target - max_neighbor)

    return [
        neighbor_mean,
        neighbor_std,
        diff_mean,
        diff_std,
        odd_ratio,
        even_ratio,
        continuity_score,
        target_to_min_diff,
        target_to_max_diff,
        float(len(arr)),
    ]


def _constraint_solving_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Advanced constraint solving features."""
    r, c = pos
    R, C = masked.shape
    max_val = R * C

    # Get candidates for this position
    candidates = _get_all_candidates(masked, pos)

    # Basic feasibility
    target_is_candidate = 1.0 if target in candidates else 0.0
    num_candidates = len(candidates)
    candidate_ratio = num_candidates / max_val

    # Uniqueness scoring - how unique is this target placement?
    empty_positions = list(zip(*np.where(masked == -1)))
    target_possible_positions = 0

    for er, ec in empty_positions:
        if (er, ec) == pos:
            continue
        pos_candidates = _get_all_candidates(masked, (er, ec))
        if target in pos_candidates:
            target_possible_positions += 1

    target_uniqueness = 1.0 / (target_possible_positions + 1)

    # Constraint propagation impact
    constraint_impact = 0.0
    if target_is_candidate > 0:
        # How many other positions would be affected by placing target here?
        temp_board = masked.copy()
        temp_board[r, c] = target

        for er, ec in empty_positions:
            if (er, ec) == pos:
                continue
            if er == r or ec == c:  # Same row or column
                old_candidates = _get_all_candidates(masked, (er, ec))
                new_candidates = _get_all_candidates(temp_board, (er, ec))
                if len(old_candidates) > len(new_candidates):
                    constraint_impact += 1.0

    constraint_impact_ratio = constraint_impact / max(1, len(empty_positions) - 1)

    return [
        target_is_candidate,
        num_candidates,
        candidate_ratio,
        target_uniqueness,
        target_possible_positions,
        constraint_impact_ratio,
    ]


def _numerical_sequence_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Features based on numerical sequences and patterns."""
    r, c = pos
    R, C = masked.shape

    # Row sequence analysis
    row = masked[r, :]
    row_known = row[row != -1]
    row_sequence_score = 0.0
    row_gap_fill_score = 0.0

    if len(row_known) > 0:
        # Check if target would create/continue arithmetic sequence
        row_with_target = row.copy()
        row_with_target[c] = target
        known_with_target = row_with_target[row_with_target != -1]

        if len(known_with_target) >= 3:
            sorted_vals = np.sort(known_with_target)
            diffs = np.diff(sorted_vals)
            if len(set(diffs)) == 1:  # Arithmetic sequence
                row_sequence_score = 1.0

        # Gap filling score - how well does target fill numerical gaps?
        if len(row_known) > 0:
            sorted_known = np.sort(row_known)
            target_pos = np.searchsorted(sorted_known, target)
            if 0 < target_pos < len(sorted_known):
                gap_before = target - sorted_known[target_pos - 1]
                gap_after = sorted_known[target_pos] - target
                if gap_before > 0 and gap_after > 0:
                    row_gap_fill_score = 1.0 / (gap_before + gap_after)

    # Column sequence analysis
    col = masked[:, c]
    col_known = col[col != -1]
    col_sequence_score = 0.0
    col_gap_fill_score = 0.0

    if len(col_known) > 0:
        col_with_target = col.copy()
        col_with_target[r] = target
        known_with_target = col_with_target[col_with_target != -1]

        if len(known_with_target) >= 3:
            sorted_vals = np.sort(known_with_target)
            diffs = np.diff(sorted_vals)
            if len(set(diffs)) == 1:
                col_sequence_score = 1.0

        if len(col_known) > 0:
            sorted_known = np.sort(col_known)
            target_pos = np.searchsorted(sorted_known, target)
            if 0 < target_pos < len(sorted_known):
                gap_before = target - sorted_known[target_pos - 1]
                gap_after = sorted_known[target_pos] - target
                if gap_before > 0 and gap_after > 0:
                    col_gap_fill_score = 1.0 / (gap_before + gap_after)

    return [
        row_sequence_score,
        col_sequence_score,
        row_gap_fill_score,
        col_gap_fill_score,
    ]


def _spatial_reasoning_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Spatial reasoning and neighborhood features."""
    r, c = pos
    R, C = masked.shape

    features = []

    # Multi-scale neighborhood analysis
    for radius in [1, 2, 3]:
        r_start = max(0, r - radius)
        r_end = min(R, r + radius + 1)
        c_start = max(0, c - radius)
        c_end = min(C, c + radius + 1)

        neighborhood = masked[r_start:r_end, c_start:c_end]

        # Neighborhood statistics
        known_vals = neighborhood[neighborhood != -1]
        if len(known_vals) > 0:
            target_count = np.sum(known_vals == target)
            target_density = target_count / len(known_vals)
            mean_val = known_vals.mean()
            target_deviation = abs(target - mean_val) / (R * C)
        else:
            target_density = 0.0
            target_deviation = 0.5

        features.extend([target_density, target_deviation])

    # Distance-based features
    known_positions = list(zip(*np.where(masked != -1)))
    if known_positions:
        distances_to_same_value = []
        distances_to_different_values = []

        for kr, kc in known_positions:
            dist = np.sqrt((r - kr) ** 2 + (c - kc) ** 2)
            if masked[kr, kc] == target:
                distances_to_same_value.append(dist)
            else:
                distances_to_different_values.append(dist)

        min_dist_same = (
            min(distances_to_same_value) if distances_to_same_value else R + C
        )
        min_dist_diff = (
            min(distances_to_different_values)
            if distances_to_different_values
            else R + C
        )

        features.extend([min_dist_same, min_dist_diff])
    else:
        features.extend([0.0, 0.0])

    return features


def _enhanced_board_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Enhanced feature extraction combining multiple reasoning approaches."""
    r, c = pos
    R, C = masked.shape

    features = []

    # Basic position features
    features.extend(
        [
            float(r) / R,
            float(c) / C,
            float(r * c) / (R * C),
            float(target) / (R * C),
            float(target % 10) / 10 if target >= 10 else 0.0,
        ]
    )

    # Add all specialized feature sets
    features.extend(_constraint_solving_features(masked, target, pos))
    features.extend(_numerical_sequence_features(masked, target, pos))
    features.extend(_spatial_reasoning_features(masked, target, pos))
    features.extend(_eight_neighbor_features(masked, target, pos))

    # Board completion features
    total_cells = R * C
    filled_cells = np.sum(masked != -1)
    completion_ratio = filled_cells / total_cells

    row_completion = np.sum(masked[r, :] != -1) / C
    col_completion = np.sum(masked[:, c] != -1) / R

    features.extend(
        [
            completion_ratio,
            row_completion,
            col_completion,
            abs(row_completion - col_completion),  # Balance between row/col completion
        ]
    )

    # Target frequency analysis
    all_known = masked[masked != -1]
    if len(all_known) > 0:
        target_frequency = np.sum(all_known == target) / len(all_known)
        expected_frequency = 1.0 / total_cells
        frequency_deviation = abs(target_frequency - expected_frequency)
    else:
        frequency_deviation = 0.0

    features.append(frequency_deviation)

    return features


def _board_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Wrapper adding duplicate and range checks to enhanced features."""
    feats = _enhanced_board_features(masked, target, pos)

    vals = masked[masked != -1]
    duplicate_count = float(np.sum(vals == target))
    feats.append(duplicate_count)

    max_val = masked.shape[0] * masked.shape[1]
    in_range = 1.0 if 1 <= target <= max_val else 0.0
    feats.append(in_range)

    return feats


def _flush(
    size: str,
    buf: list[Tuple[List[float], int, int]],
    out_root: Path,
    cnt: dict[str, int],
) -> None:
    """Flush buffer to disk."""
    if not buf:
        return
    X = np.asarray([x for x, _, _ in buf], dtype=np.float32)
    y = np.asarray([y for _, y, _ in buf], dtype=np.uint8)
    bid = np.asarray([b for _, _, b in buf], dtype=np.int32)
    dest = out_root / size
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"part_{cnt[size]:04d}.npz", X=X, y=y, bid=bid)
    cnt[size] += 1
    buf.clear()


def _apply_random_mask(
    board: np.ndarray, ratio: float, rng: random.Random
) -> Tuple[np.ndarray, list[Tuple[int, int, int]]]:
    """Apply random masking to board."""
    if ratio <= 0:
        return board, []
    r, c = board.shape
    cells = [(i, j) for i in range(r) for j in range(c)]
    rng.shuffle(cells)
    n = max(1, int(round(len(cells) * ratio)))
    masked = board.copy()
    info: list[Tuple[int, int, int]] = []
    for i, j in cells[:n]:
        info.append((i, j, int(masked[i, j])))
        masked[i, j] = -1
    return masked, info


def extract_features(
    root: Path,
    out_feat: Path,
    shard_size: int,
    workers: int,
    mask_ratio: float,
    mask_range: tuple[float, float] | None,
) -> None:
    """Extract features from board data."""
    buf: dict[str, list[Tuple[List[float], int, int]]] = defaultdict(list)
    cnt: dict[str, int] = defaultdict(int)
    board_idx: dict[str, int] = defaultdict(int)

    for sd in out_feat.iterdir():
        if not sd.is_dir():
            continue
        existing = sorted(sd.glob("part_*.npz"))
        if existing:
            cnt[sd.name] = len(existing)

    rng = random.Random(42)

    for board, target in tqdm(iter_objects(root), unit="board"):
        R, C = board.shape
        key = f"{R}x{C}"
        bid = board_idx[key]
        board_idx[key] += 1

        if target is not None:
            pos = tuple(zip(*np.where(board == -1)))[0]
            for r in range(R):
                for c in range(C):
                    lbl = 1 if (r, c) == pos else 0
                    buf[key].append(
                        (_board_features(board, int(target), (r, c)), lbl, bid)
                    )
        else:
            ratio = mask_ratio
            if mask_range is not None:
                lo, hi = mask_range
                ratio = rng.uniform(min(lo, hi), max(lo, hi))

            if ratio > 0:
                masked, slots = _apply_random_mask(board, ratio, rng)
                coords = [(rr, cc) for rr in range(R) for cc in range(C)]

                for mr, mc, tv in slots:
                    buf[key].append((_board_features(masked, tv, (mr, mc)), 1, bid))
                    negs = [p for p in coords if p != (mr, mc)]
                    rng.shuffle(negs)
                    for rr, cc in negs[:NEG_RATIO]:
                        buf[key].append((_board_features(masked, tv, (rr, cc)), 0, bid))
            else:
                for r in range(R):
                    for c in range(C):
                        tv = int(board[r, c])
                        masked = board.copy()
                        masked[r, c] = -1
                        buf[key].append((_board_features(masked, tv, (r, c)), 1, bid))
                        negs = [
                            (rr, cc)
                            for rr in range(R)
                            for cc in range(C)
                            if (rr, cc) != (r, c)
                        ]
                        rng.shuffle(negs)
                        for rr, cc in negs[:NEG_RATIO]:
                            buf[key].append(
                                (_board_features(masked, tv, (rr, cc)), 0, bid)
                            )

        if len(buf[key]) >= shard_size:
            _flush(key, buf[key], out_feat, cnt)

    for k in list(buf.keys()):
        _flush(k, buf[k], out_feat, cnt)


def _ensure_labels(
    X: np.ndarray, y: np.ndarray, bid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure both classes are present."""
    if np.all(y == 0) or np.all(y == 1):
        X_fix = X[:1].copy()
        y_fix = np.array([1 - y[0]], dtype=y.dtype)
        bid_fix = np.array([bid[0]], dtype=bid.dtype)
        X = np.vstack([X, X_fix])
        y = np.concatenate([y, y_fix])
        bid = np.concatenate([bid, bid_fix])
    return X, y, bid


def train_models(out_feat: Path, out_model: Path, trees_per: int, workers: int) -> None:
    """Train LightGBM models with improved parameters."""
    # Enhanced parameters for better performance
    params = dict(
        objective="binary",
        learning_rate=0.03,  # Lower learning rate
        num_leaves=64,  # Increased complexity
        max_depth=8,  # Increased depth
        feature_fraction=0.9,  # Use more features
        bagging_fraction=0.85,
        min_data_in_leaf=10,  # Reduced for more granular learning
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        metric=["binary_logloss", "auc"],
        verbosity=-1,
        seed=42,
        boost_from_average=False,  # Important for imbalanced data
    )

    out_model.mkdir(exist_ok=True)

    for sd in sorted(out_feat.iterdir()):
        if not sd.is_dir():
            continue

        npzs = sorted(sd.glob("part_*.npz"))
        if not npzs:
            continue

        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_bid: list[np.ndarray] = []

        for npz in npzs:
            data = np.load(npz)
            X, y, bid = _ensure_labels(data["X"], data["y"], data["bid"])
            all_X.append(X)
            all_y.append(y)
            all_bid.append(bid)

        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)
        bid_all = np.concatenate(all_bid)

        uniq = np.unique(bid_all)
        if uniq.size < 2:
            train_b = valid_b = uniq
        else:
            train_b, valid_b = train_test_split(uniq, test_size=0.2, random_state=42)

        train_mask = np.isin(bid_all, train_b)
        valid_mask = ~train_mask

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_valid = X_all[valid_mask]
        y_valid = y_all[valid_mask]

        if X_valid.size == 0:
            X_valid = X_train
            y_valid = y_train
            valid_b = train_b

        scaler = Float32StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        train_ds = lgb.Dataset(X_train, y_train)
        valid_ds = lgb.Dataset(X_valid, y_valid, reference=train_ds)

        # Fixed: Use either scale_pos_weight OR is_unbalance, not both
        params.update(
            {
                "num_threads": workers,
                "scale_pos_weight": scale_pos_weight,
                # Remove is_unbalance since we're using scale_pos_weight
            }
        )

        booster = lgb.train(
            params,
            train_ds,
            num_boost_round=trees_per,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(100)],  # More patience
        )

        joblib.dump({"model": booster, "scaler": scaler}, out_model / f"{sd.name}.pkl")

        preds = booster.predict(X_valid, num_iteration=booster.best_iteration)
        auc = roc_auc_score(y_valid, preds)
        acc = accuracy_score(y_valid, preds > 0.5)
        f1 = f1_score(y_valid, preds > 0.5)

        def top_k_hit_rate(k: int) -> float:
            hit = 0
            total = 0
            for b in valid_b:
                idx = np.where(bid_all == b)[0]
                lbls = y_all[idx]
                p = booster.predict(
                    scaler.transform(X_all[idx]), num_iteration=booster.best_iteration
                )
                pos_indices = np.where(lbls == 1)[0]
                if len(pos_indices) == 0:
                    continue

                pos_idx = idx[pos_indices[0]]  # Get actual position
                top_k_indices = idx[np.argsort(-p)[:k]]
                if pos_idx in top_k_indices:
                    hit += 1
                total += 1
            return hit / total if total else 0.0

        hit1 = top_k_hit_rate(1)
        hit3 = top_k_hit_rate(3)
        hit5 = top_k_hit_rate(5)

        print(
            f"✔ {sd.name} | trees={booster.best_iteration} | AUC={auc:.3f} | "
            f"acc={acc:.3f} | f1={f1:.3f} | hit@1={hit1:.3f} | hit@3={hit3:.3f} | hit@5={hit5:.3f}"
        )


def main(argv: List[str] | None = None) -> None:
    """Main function."""
    pa = argparse.ArgumentParser(
        description="Enhanced LightGBM board completion pipeline"
    )
    pa.add_argument("--root", default=".", help="Data root directory")
    pa.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    pa.add_argument("--trees-per-shard", type=int, default=TREES_PER)
    pa.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    pa.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", 8)),
        help="Number of CPU threads to use",
    )
    pa.add_argument("--train-only", action="store_true", help="Skip feature extraction")
    pa.add_argument("--out-feat", default="features")
    pa.add_argument("--out-model", default="models")
    pa.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Default mask ratio for full boards",
    )
    pa.add_argument(
        "--mask-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Random mask ratio range for data augmentation",
    )
    args = pa.parse_args(argv)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    root = Path(args.root)
    out_feat = Path(args.out_feat)
    out_model = Path(args.out_model)

    if not args.train_only:
        out_feat.mkdir(exist_ok=True)
        print("\n🔍 Extracting enhanced features …")
        extract_features(
            root,
            out_feat,
            args.shard_size,
            args.workers,
            args.mask_ratio,
            tuple(args.mask_range) if args.mask_range else None,
        )

    print("\n🏋️  Training enhanced models …")
    train_models(out_feat, out_model, args.trees_per_shard, args.workers)
    print("✅ Done. Enhanced models in", out_model)


if __name__ == "__main__":
    main()
