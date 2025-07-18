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
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


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


def _board_features(
    masked: np.ndarray, target: int, pos: Tuple[int, int]
) -> List[float]:
    r, c = pos
    row_sum = np.where(masked != -1, masked, 0).sum(axis=1)
    col_sum = np.where(masked != -1, masked, 0).sum(axis=0)
    m3, v3, rg3 = _local_stats(masked, r, c, 1)
    m5, v5, rg5 = _local_stats(masked, r, c, 2)
    feats = [
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
    feats.extend(_svd_feats(masked))
    vals = masked[masked != -1]
    if vals.size:
        board_min = float(vals.min())
        board_max = float(vals.max())
        board_range = board_max - board_min
    else:
        board_min = board_max = board_range = 0.0
    feats.extend([board_min, board_max, board_range, rg3, rg5])
    return feats


def _flush(
    size: str, buf: list[Tuple[List[float], int]], out_root: Path, cnt: dict[str, int]
) -> None:
    if not buf:
        return
    X = np.asarray([x for x, _ in buf], dtype=np.float32)
    y = np.asarray([y for _, y in buf], dtype=np.uint8)
    dest = out_root / size
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"part_{cnt[size]:04d}.npz", X=X, y=y)
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
    buf: dict[str, list[Tuple[List[float], int]]] = defaultdict(list)
    cnt: dict[str, int] = defaultdict(int)

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
        if target is not None:
            pos = tuple(zip(*np.where(board == -1)))[0]
            for r in range(R):
                for c in range(C):
                    lbl = 1 if (r, c) == pos else 0
                    buf[key].append((_board_features(board, int(target), (r, c)), lbl))
        else:
            ratio = mask_ratio
            if mask_range is not None:
                lo, hi = mask_range
                ratio = rng.uniform(min(lo, hi), max(lo, hi))
            if ratio > 0:
                masked, slots = _apply_random_mask(board, ratio, rng)
                coords = [(rr, cc) for rr in range(R) for cc in range(C)]
                for mr, mc, tv in slots:
                    buf[key].append((_board_features(masked, tv, (mr, mc)), 1))
                    negs = [p for p in coords if p != (mr, mc)]
                    rng.shuffle(negs)
                    for rr, cc in negs[:NEG_RATIO]:
                        buf[key].append((_board_features(masked, tv, (rr, cc)), 0))
            else:
                for r in range(R):
                    for c in range(C):
                        tv = int(board[r, c])
                        masked = board.copy()
                        masked[r, c] = -1
                        buf[key].append((_board_features(masked, tv, (r, c)), 1))
                        negs = [
                            (rr, cc)
                            for rr in range(R)
                            for cc in range(C)
                            if (rr, cc) != (r, c)
                        ]
                        rng.shuffle(negs)
                        for rr, cc in negs[:NEG_RATIO]:
                            buf[key].append((_board_features(masked, tv, (rr, cc)), 0))
        if len(buf[key]) >= shard_size:
            _flush(key, buf[key], out_feat, cnt)
    for k in list(buf.keys()):
        _flush(k, buf[k], out_feat, cnt)


def _ensure_labels(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.all(y == 0) or np.all(y == 1):
        X_fix = X[:1].copy()
        y_fix = np.array([1 - y[0]], dtype=y.dtype)
        X = np.vstack([X, X_fix])
        y = np.concatenate([y, y_fix])
    return X, y


def train_models(out_feat: Path, out_model: Path, trees_per: int) -> None:
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
        booster = None
        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for npz in npzs:
            data = np.load(npz)
            X, y = _ensure_labels(data["X"], data["y"])
            ds = lgb.Dataset(X, y)
            booster = lgb.train(
                params,
                ds,
                num_boost_round=trees_per,
                init_model=booster,
                keep_training_booster=True,
            )
            all_X.append(X)
            all_y.append(y)
        joblib.dump(booster, out_model / f"{sd.name}.pkl")
        X_all = np.concatenate(all_X)
        y_all = np.concatenate(all_y)
        preds = booster.predict(X_all)
        auc = roc_auc_score(y_all, preds)
        acc = accuracy_score(y_all, preds > 0.5)
        f1 = f1_score(y_all, preds > 0.5)
        print(
            f"✔ {sd.name} | trees={booster.num_trees()} | AUC={auc:.3f} | acc={acc:.3f} | f1={f1:.3f}"
        )


def main(argv: List[str] | None = None) -> None:
    pa = argparse.ArgumentParser(description="LightGBM offline training pipeline")
    pa.add_argument("--root", default=".", help="Data root directory")
    pa.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    pa.add_argument("--trees-per-shard", type=int, default=TREES_PER)
    pa.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
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
    train_models(out_feat, out_model, args.trees_per_shard)
    print("✅ Done. Models in", out_model)


if __name__ == "__main__":  # pragma: no cover
    main()
