#!/usr/bin/env python
"""All-in-one LightGBM training pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Tuple

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
except Exception:  # noqa: W0703 - optional dependency

    def tqdm(x, **kwargs):  # type: ignore
        """Fallback if tqdm is missing."""
        return x


NEG_RATIO = 3
SHARD_SIZE = 100_000
TREES_PER = 200

# limit threads
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")


def _yield_json(fp) -> Iterator[object]:
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


def _local_stats(mat: np.ndarray, r: int, c: int, k: int) -> Tuple[float, float, float]:
    """Return mean, variance and range within a square window."""
    sub = mat[max(0, r - k) : r + k + 1, max(0, c - k) : c + k + 1].ravel()
    sub = sub[sub != -1]
    if sub.size:
        return float(sub.mean()), float(sub.var()), float(sub.max() - sub.min())
    return 0.0, 0.0, 0.0


def _svd_feats(masked: np.ndarray) -> List[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filled = np.where(masked == -1, masked.mean(), masked)
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
    approx = (u[:, :4] @ np.diag(s[:4]) @ vt[:4, :]).flatten()
    if approx.size < 6:
        approx = np.pad(approx, (0, 6 - approx.size))
    return approx[:6].tolist()


def _exclusion_method_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Sudoku-style exclusion logic features."""
    r, c = pos
    R, C = masked.shape
    total_nums = R * C

    row_used = set(masked[r, :][masked[r, :] != -1])
    col_used = set(masked[:, c][masked[:, c] != -1])
    global_used = set(masked[masked != -1].flatten())

    row_feasible = 1.0 if target not in row_used else 0.0
    col_feasible = 1.0 if target not in col_used else 0.0
    global_feasible = 1.0 if target not in global_used else 0.0
    overall_feasible = row_feasible * col_feasible * global_feasible

    all_nums = set(range(1, total_nums + 1))
    row_candidates = all_nums - row_used
    col_candidates = all_nums - col_used
    global_candidates = all_nums - global_used
    position_candidates = row_candidates & col_candidates & global_candidates

    num_position_candidates = len(position_candidates)
    candidate_density = num_position_candidates / total_nums

    target_competition_score = 0.0
    if target in global_candidates:
        empty_positions = list(zip(*np.where(masked == -1)))
        feasible_positions = 0
        for er, ec in empty_positions:
            if er == r and ec == c:
                continue
            row_ok = target not in set(masked[er, :][masked[er, :] != -1])
            col_ok = target not in set(masked[:, ec][masked[:, ec] != -1])
            if row_ok and col_ok:
                feasible_positions += 1
        total_empty = len(empty_positions) - 1
        if total_empty > 0:
            target_competition_score = 1.0 - (feasible_positions / total_empty)

    return [
        row_feasible,
        col_feasible,
        global_feasible,
        overall_feasible,
        num_position_candidates,
        candidate_density,
        target_competition_score,
    ]


def _naked_singles_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Features describing naked singles."""
    r, c = pos
    R, C = masked.shape
    total_nums = R * C

    all_nums = set(range(1, total_nums + 1))
    row_used = set(masked[r, :][masked[r, :] != -1])
    col_used = set(masked[:, c][masked[:, c] != -1])
    global_used = set(masked[masked != -1].flatten())

    candidates = all_nums - row_used - col_used - global_used

    is_naked_single = 1.0 if len(candidates) == 1 else 0.0
    target_is_only_choice = (
        1.0 if len(candidates) == 1 and target in candidates else 0.0
    )

    candidates_count = len(candidates)
    naked_single_proximity = 1.0 / (candidates_count + 1)

    return [
        is_naked_single,
        target_is_only_choice,
        candidates_count,
        naked_single_proximity,
    ]


def _hidden_singles_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Features for hidden singles in row/column."""
    r, c = pos
    R, C = masked.shape

    row_empty_cols = [j for j in range(C) if masked[r, j] == -1]
    row_target_positions = []
    for j in row_empty_cols:
        col_used = set(masked[:, j][masked[:, j] != -1])
        if target not in col_used:
            row_target_positions.append(j)

    row_hidden_single = (
        1.0 if len(row_target_positions) == 1 and c in row_target_positions else 0.0
    )

    col_empty_rows = [i for i in range(R) if masked[i, c] == -1]
    col_target_positions = []
    for i in col_empty_rows:
        row_used = set(masked[i, :][masked[i, :] != -1])
        if target not in row_used:
            col_target_positions.append(i)

    col_hidden_single = (
        1.0 if len(col_target_positions) == 1 and r in col_target_positions else 0.0
    )

    row_target_density = (
        len(row_target_positions) / len(row_empty_cols) if row_empty_cols else 0.0
    )
    col_target_density = (
        len(col_target_positions) / len(col_empty_rows) if col_empty_rows else 0.0
    )

    return [
        row_hidden_single,
        col_hidden_single,
        len(row_target_positions),
        len(col_target_positions),
        row_target_density,
        col_target_density,
    ]


def _forced_chain_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Forced chain logical inference features."""
    r, c = pos
    R, C = masked.shape

    forced_moves = 0
    contradiction_risk = 0.0

    temp_board = masked.copy()
    temp_board[r, c] = target

    empty_positions = list(zip(*np.where(masked == -1)))
    if (r, c) in empty_positions:
        empty_positions.remove((r, c))

    for er, ec in empty_positions:
        all_nums = set(range(1, R * C + 1))
        row_used = set(temp_board[er, :][temp_board[er, :] != -1])
        col_used = set(temp_board[:, ec][temp_board[:, ec] != -1])

        candidates = all_nums - row_used - col_used

        if len(candidates) == 1:
            forced_moves += 1
        elif len(candidates) == 0:
            contradiction_risk = 1.0
            break

    forced_moves_ratio = forced_moves / len(empty_positions) if empty_positions else 0.0

    return [forced_moves, forced_moves_ratio, contradiction_risk]


def _symmetry_pattern_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Symmetry and pattern based features."""
    r, c = pos
    R, C = masked.shape

    center_r, center_c = R / 2, C / 2
    center_distance = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
    normalized_center_dist = center_distance / np.sqrt(center_r**2 + center_c**2)

    main_diagonal = 1.0 if r == c else 0.0
    anti_diagonal = 1.0 if r + c == min(R, C) - 1 else 0.0

    known_positions = list(zip(*np.where(masked != -1)))
    if len(known_positions) > 2:
        values = []
        distances_to_corner = []
        for kr, kc in known_positions:
            val = masked[kr, kc]
            dist = np.sqrt(kr**2 + kc**2)
            values.append(val)
            distances_to_corner.append(dist)

        if len(set(distances_to_corner)) > 1:
            position_value_correlation = np.corrcoef(values, distances_to_corner)[0, 1]
            if np.isnan(position_value_correlation):
                position_value_correlation = 0.0
        else:
            position_value_correlation = 0.0

        target_distance = np.sqrt(r**2 + c**2)
        if values and distances_to_corner:
            expected_value_by_pattern = np.mean(values) + position_value_correlation * (
                target_distance - np.mean(distances_to_corner)
            ) * np.std(values) / (np.std(distances_to_corner) + 1e-8)
            pattern_consistency = 1.0 - abs(target - expected_value_by_pattern) / (
                R * C
            )
        else:
            pattern_consistency = 0.0
    else:
        position_value_correlation = 0.0
        pattern_consistency = 0.0

    return [
        normalized_center_dist,
        main_diagonal,
        anti_diagonal,
        position_value_correlation,
        pattern_consistency,
    ]


def _constraint_propagation_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Advanced constraint propagation features."""
    r, c = pos
    R, C = masked.shape
    total_nums = R * C

    all_nums = set(range(1, total_nums + 1))
    global_used = set(masked[masked != -1].flatten())
    remaining_nums = all_nums - global_used

    if target in remaining_nums:
        empty_positions = list(zip(*np.where(masked == -1)))
        target_feasible_positions = 0
        for er, ec in empty_positions:
            row_ok = target not in set(masked[er, :][masked[er, :] != -1])
            col_ok = target not in set(masked[:, ec][masked[:, ec] != -1])
            if row_ok and col_ok:
                target_feasible_positions += 1
        target_scarcity = 1.0 / (target_feasible_positions + 1)
    else:
        target_scarcity = 0.0

    empty_positions = list(zip(*np.where(masked == -1)))
    if (r, c) in empty_positions:
        empty_positions.remove((r, c))

    constraint_impact = 0
    for er, ec in empty_positions:
        if er == r or ec == c:
            row_used = set(masked[er, :][masked[er, :] != -1])
            col_used = set(masked[:, ec][masked[:, ec] != -1])
            candidates = all_nums - row_used - col_used - global_used
            if target in candidates:
                constraint_impact += 1

    normalized_constraint_impact = (
        constraint_impact / len(empty_positions) if empty_positions else 0.0
    )

    row_completion = np.sum(masked[r, :] != -1) / C
    col_completion = np.sum(masked[:, c] != -1) / R
    local_completion = (row_completion + col_completion) / 2

    return [
        target_scarcity,
        normalized_constraint_impact,
        row_completion,
        col_completion,
        local_completion,
    ]


def _calculate_gap_score(known_values: np.ndarray, target: int, max_val: int) -> float:
    """Score how well the target fills numerical gaps."""
    if len(known_values) == 0:
        return 0.5

    sorted_known = np.sort(known_values)

    insert_pos = np.searchsorted(sorted_known, target)

    if insert_pos == 0:
        gap_size = sorted_known[0] - 1
    elif insert_pos == len(sorted_known):
        gap_size = max_val - sorted_known[-1]
    else:
        gap_size = sorted_known[insert_pos] - sorted_known[insert_pos - 1] - 1

    return min(1.0, gap_size / max_val)


def _statistical_inference_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Statistical inference features based on board distribution."""
    r, c = pos
    R, C = masked.shape

    row_known = masked[r, :][masked[r, :] != -1]
    col_known = masked[:, c][masked[:, c] != -1]

    if len(row_known) > 0:
        row_mean = float(row_known.mean())
        row_std = float(row_known.std()) if len(row_known) > 1 else 0.0
        row_gap_score = _calculate_gap_score(row_known, target, R * C)
    else:
        row_mean = row_std = row_gap_score = 0.0

    if len(col_known) > 0:
        col_mean = float(col_known.mean())
        col_std = float(col_known.std()) if len(col_known) > 1 else 0.0
        col_gap_score = _calculate_gap_score(col_known, target, R * C)
    else:
        col_mean = col_std = col_gap_score = 0.0

    global_known = masked[masked != -1]
    if len(global_known) > 0:
        global_mean = float(global_known.mean())
        global_std = float(global_known.std()) if len(global_known) > 1 else 0.0
        target_z_score = (target - global_mean) / (global_std + 1e-8)
        target_percentile = (global_known < target).mean()
    else:
        global_mean = global_std = 0.0
        target_z_score = target_percentile = 0.0

    return [
        row_mean / (R * C),
        row_std / (R * C),
        col_mean / (R * C),
        col_std / (R * C),
        global_mean / (R * C),
        global_std / (R * C),
        target_z_score,
        target_percentile,
        row_gap_score,
        col_gap_score,
    ]


def comprehensive_excel_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Aggregate advanced features for the board."""

    features: List[float] = []
    features.extend(_exclusion_method_features(masked, target, pos))
    features.extend(_naked_singles_features(masked, target, pos))
    features.extend(_hidden_singles_features(masked, target, pos))
    features.extend(_forced_chain_features(masked, target, pos))
    features.extend(_constraint_propagation_features(masked, target, pos))
    features.extend(_symmetry_pattern_features(masked, target, pos))
    features.extend(_statistical_inference_features(masked, target, pos))

    r, c = pos
    R, C = masked.shape
    basic_features = [
        float(r / R),
        float(c / C),
        float(r * c / (R * C)),
        float(target / (R * C)),
        float(target % 10) / 10,
        float(target // 10) / (R * C // 10 + 1),
    ]
    features.extend(basic_features)
    return features


def enhanced_board_features_v2(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    """Wrapper for the comprehensive feature extractor."""
    return comprehensive_excel_features(masked, target, pos)


def _board_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    r, c = pos
    row_sum = np.where(masked != -1, masked, 0).sum(axis=1)
    col_sum = np.where(masked != -1, masked, 0).sum(axis=0)
    m3, v3, rg3 = _local_stats(masked, r, c, 1)
    m5, v5, rg5 = _local_stats(masked, r, c, 2)

    feats = comprehensive_excel_features(masked, target, pos)

    basic = [
        float(r),
        float(c),
        float(r * c),
        float(r**2),
        float(c**2),
        m3,
        v3,
        m5,
        v5,
        float(row_sum[r]),
        float(col_sum[c]),
        float(target),
        float(target % 10),
        float(target // 10),
    ]
    feats.extend(basic)
    feats.extend(_svd_feats(masked))

    vals = masked[masked != -1]
    if vals.size:
        board_min = float(vals.min())
        board_max = float(vals.max())
        board_range = board_max - board_min
    else:
        board_min = board_max = board_range = 0.0
    feats.extend([board_min, board_max, board_range, rg3, rg5])

    vals = masked[masked != -1]
    count_duplicate = float(np.sum(vals == target))
    feats.append(count_duplicate)

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
    if np.all(y == 0) or np.all(y == 1):
        X_fix = X[:1].copy()
        y_fix = np.array([1 - y[0]], dtype=y.dtype)
        bid_fix = np.array([bid[0]], dtype=bid.dtype)
        X = np.vstack([X, X_fix])
        y = np.concatenate([y, y_fix])
        bid = np.concatenate([bid, bid_fix])
    return X, y, bid


def train_models(out_feat: Path, out_model: Path, trees_per: int, workers: int) -> None:
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=48,
        max_depth=7,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        min_data_in_leaf=20,
        metric=["binary_logloss"],
        verbosity=-1,
        seed=42,
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
        scale_pos_weight = neg / pos if pos else 1.0

        train_ds = lgb.Dataset(X_train, y_train)
        valid_ds = lgb.Dataset(X_valid, y_valid, reference=train_ds)

        params.update({"num_threads": workers, "scale_pos_weight": scale_pos_weight})

        booster = lgb.train(
            params,
            train_ds,
            num_boost_round=trees_per,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(50)],
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
                pos_idx = idx[np.argmax(lbls)]
                rank = idx[np.argsort(-p)][:k]
                if pos_idx in rank:
                    hit += 1
                total += 1
            return hit / total if total else 0.0

        hit1 = top_k_hit_rate(1)
        hit3 = top_k_hit_rate(3)
        print(
            f"✔ {sd.name} | trees={booster.best_iteration} | AUC={auc:.3f} | acc={acc:.3f} | f1={f1:.3f} | hit@1={hit1:.3f} | hit@3={hit3:.3f}"
        )


def main(argv: List[str] | None = None) -> None:
    pa = argparse.ArgumentParser(description="LightGBM offline training pipeline")
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
        "--mask-ratio", type=float, default=0.0, help="Mask ratio for full boards"
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
        print("\n🔍 Extracting features …")
        extract_features(
            root,
            out_feat,
            args.shard_size,
            args.workers,
            args.mask_ratio,
            tuple(args.mask_range) if args.mask_range else None,
        )

    print("\n🏋️  Training models …")
    train_models(out_feat, out_model, args.trees_per_shard, args.workers)
    print("✅ Done. Models in", out_model)


if __name__ == "__main__":  # pragma: no cover
    main()
