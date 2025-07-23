#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chunked LightGBM training from NPZ shards.
- 固定一份大型驗證集 (valid_rows)，用 early_stopping callback 控制每次 train。
- 每個 chunk 續訓 (init_model + keep_training_booster)。
- 跨 chunk 比較 global_best_auc，連續 chunk 沒進步就整體停掉。
- free_raw_data=False on valid set，避免續訓時報錯。
"""

import argparse
import os
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import joblib
import lightgbm as lgb
import numpy as np


def tick(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_size_dirs(root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for p in root.iterdir():
        if p.is_dir() and re.match(r"^\d+x\d+$", p.name):
            files = sorted(p.glob("part_*.npz"))
            if files:
                out[p.name] = files
    return out


def load_npz(fp: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(fp)
    X = d["X"].astype(np.float32, copy=False)
    y = d["y"].astype(np.int8, copy=False)
    return X, y


def iter_chunks(files: List[Path], batch: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    bufX, bufy = [], []
    for i, f in enumerate(files, 1):
        X, y = load_npz(f)
        bufX.append(X); bufy.append(y)
        if i % batch == 0:
            yield np.vstack(bufX), np.concatenate(bufy)
            bufX.clear(); bufy.clear()
    if bufX:
        yield np.vstack(bufX), np.concatenate(bufy)


def make_valid_set(files: List[Path], target_rows: int, seed: int
                   ) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    take, rows = [], 0
    for f in shuffled:
        X, y = load_npz(f)
        take.append((X, y, f))
        rows += len(y)
        if rows >= target_rows:
            break

    Xv = np.vstack([t[0] for t in take])
    yv = np.concatenate([t[1] for t in take])
    used = {t[2] for t in take}
    rest = [f for f in shuffled if f not in used]
    return Xv, yv, rest


def deepcopy_booster(bst: lgb.Booster) -> lgb.Booster:
    """LightGBM Booster 沒有原生 deepcopy，轉字串再回讀。"""
    return lgb.Booster(model_str=bst.model_to_string())


def train_one_size(size_key: str, shard_files: List[Path], models_dir: Path,
                   args: argparse.Namespace) -> None:
    tick(f"=== [{size_key}] shards={len(shard_files)} ===")

    # 驗證集固定
    Xv, yv, train_files = make_valid_set(shard_files, args.valid_rows, args.seed)
    tick(f"[{size_key}] valid set: {Xv.shape}")
    dvalid = lgb.Dataset(Xv, yv, free_raw_data=False)  # 續訓必須保留 raw data

    # 參數
    params = dict(
        objective=args.objective,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        max_bin=255,
        metric=["auc", "binary_logloss"] if args.objective == "binary" else "rmse",
        num_threads=args.threads,
        verbose=-1,
        seed=args.seed,
        deterministic=True,
    )

    rng = random.Random(args.seed)
    rng.shuffle(train_files)

    bst = None
    first = True
    global_best_auc = -1.0
    global_best_bst = None
    bad_chunk = 0

    for Xc, yc in iter_chunks(train_files, args.batch):
        tick(f"[{size_key}] chunk {Xc.shape} training…")
        dtrain = lgb.Dataset(Xc, yc, free_raw_data=True, reference=dvalid)

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=args.trees_first if first else args.trees_each,
            init_model=bst,
            keep_training_booster=True,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(args.early_stop, verbose=True),
                lgb.log_evaluation(10),
            ],
        )
        first = False

        cur_auc = bst.best_score["valid"].get("auc", float("nan"))
        tick(f"[{size_key}] chunk done. best_iter={bst.best_iteration}, valid_auc={cur_auc:.4f}")

        # 跨 chunk 早停 & checkpoint
        if cur_auc > global_best_auc + 1e-6:
            global_best_auc = cur_auc
            global_best_bst = deepcopy_booster(bst)
            bad_chunk = 0
            tick(f"[{size_key}] 🌟 New global best AUC={global_best_auc:.4f}")
        else:
            bad_chunk += 1
            tick(f"[{size_key}] No improvement for {bad_chunk} chunk(s)")
            if bad_chunk >= args.chunk_patience:
                tick(f"[{size_key}] 🔚 Stop early (no chunk improvement)")
                break

    # 存最佳模型
    models_dir.mkdir(parents=True, exist_ok=True)
    final_bst = global_best_bst or bst

    txt_path = models_dir / f"{size_key}_lgbm_best.txt"
    pkl_path = models_dir / f"{size_key}_lgbm_best.pkl"

    tick(f"[{size_key}] save txt → {txt_path}")
    final_bst.save_model(str(txt_path), num_iteration=final_bst.best_iteration)

    tick(f"[{size_key}] save pkl (no-compress) → {pkl_path}")
    joblib.dump(final_bst, pkl_path, compress=0)

    tick(f"[{size_key}] ✅ Done (best_auc={global_best_auc:.4f})")


def main():
    ap = argparse.ArgumentParser("Chunked LightGBM training with global early stop")
    ap.add_argument("--feat-root", type=Path, required=True, help="features 根目錄")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--batch", type=int, default=10, help="每批讀幾個 shard")
    ap.add_argument("--valid-rows", type=int, default=100_000)
    ap.add_argument("--trees-first", type=int, default=400)
    ap.add_argument("--trees-each", type=int, default=80)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--chunk-patience", type=int, default=2,
                    help="連續多少 chunk 沒進步就整體停掉")
    ap.add_argument("--objective", choices=["binary", "regression"], default="binary")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    size_map = find_size_dirs(args.feat_root)
    if not size_map:
        raise SystemExit(f"在 {args.feat_root} 找不到 'NxM' 目錄或 part_*.npz")

    tick(f"found: {', '.join(size_map.keys())}")
    for size_key, files in size_map.items():
        train_one_size(size_key, files, args.models_dir, args)

    tick("ALL DONE 🎉")


if __name__ == "__main__":
    main()
