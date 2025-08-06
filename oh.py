#!/usr/bin/env python3
"""
讀取記憶庫檔，統計「目標數字 → 同列/同行出現的數字」機率表。

輸出：
  {rows}x{cols}_nbr_counts.npy  int32   (shape = [num_targets, num_values])
  {rows}x{cols}_nbr_probs.npy   float32 (同上，已除以總和 → row-wise 機率)

適用兩種來源：
1. 舊式  {rows}x{cols}_memory_part?.npz
2. 新式  {rows}x{cols}_{keys|vals|targets|boards}_p?.npy
   （只需 targets 與 boards；若無 boards 會拋錯）
"""

from __future__ import annotations
from pathlib import Path
import re, numpy as np, typing as _t, itertools, math, sys

# ---------- 可視需要自行調整 ---------- #
DATA_DIR = Path("data_archives")   # 記憶庫資料夾
BLANK_VALUE = -1                  # 盤面中代表空格的值
# -------------------------------------- #

PAT_NPZ   = re.compile(r"(?P<tag>\d+x\d+)_memory_part\d+\.npz$")
PAT_SHARD = re.compile(r"(?P<tag>\d+x\d+)_(?P<kind>\w+)_p\d+\.npy$")

def _collect_arrays(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """
    讀入指定尺寸的 targets 與 boards，若有多個 shard 自動 concat。
    回傳 (targets, boards)：
      targets: int16, shape = [N]
      boards : int8 , shape = [N, rows*cols]
    """
    # --- 舊 npz ---
    t_lst, b_lst = [], []
    for p in sorted(DATA_DIR.glob(f"{tag}_memory_part*.npz")):
        with np.load(p) as z:
            t_lst.append(z["targets"].astype(np.int16, copy=False))
            if "boards" not in z.files:
                raise RuntimeError(f"{p.name} 缺少 boards 陣列，無法統計鄰居！")
            b_lst.append(z["boards"].astype(np.int8,  copy=False))
    # --- 新 shard ---
    tgt_shards = sorted(DATA_DIR.glob(f"{tag}_targets_p*.npy"))
    brd_shards = sorted(DATA_DIR.glob(f"{tag}_boards_p*.npy"))
    if tgt_shards and brd_shards:               # 防止重複加進來
        t_lst += [np.load(p, mmap_mode="r").astype(np.int16, copy=False)
                  for p in tgt_shards]
        b_lst += [np.load(p, mmap_mode="r").astype(np.int8 , copy=False)
                  for p in brd_shards]

    if not t_lst:
        raise RuntimeError(f"找不到 {tag} 的 targets/boards 資料")
    targets = np.concatenate(t_lst)
    boards  = np.concatenate(b_lst)
    return targets, boards

def _build_neighbor_stats(
    targets: np.ndarray,
    boards : np.ndarray,
    rows: int, cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    依照 (rows, cols) 統計鄰居。
    回傳 (counts, probs)，shape = [num_targets, num_values]。
    """
    num_values  = boards.max()          # 盤面最大值 (假設連續 1..V)
    num_targets = targets.max()
    counts = np.zeros((num_targets+1, num_values+1), dtype=np.int32)

    for brd, tgt in zip(boards, targets):
        grid = brd.reshape(rows, cols)
        pos  = np.where(grid == tgt)
        if pos[0].size == 0:
            continue                     # 目標數字沒在盤面上
        r, c = int(pos[0][0]), int(pos[1][0])
        neigh = set(grid[r, :]) | set(grid[:, c])
        neigh.discard(tgt); neigh.discard(BLANK_VALUE)
        for n in neigh:
            counts[tgt, n] += 1

    # 機率：對每個 target row 做 normalize（若全零保持零）
    probs = counts.astype(np.float32)
    row_sums = probs.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze() > 0
    probs[nonzero] /= row_sums[nonzero]
    return counts, probs

def convert_one(tag: str) -> None:
    rows, cols = map(int, tag.split("x"))
    targets, boards = _collect_arrays(tag)
    counts, probs   = _build_neighbor_stats(targets, boards, rows, cols)

    np.save(DATA_DIR / f"{tag}_nbr_counts.npy", counts)
    np.save(DATA_DIR / f"{tag}_nbr_probs.npy" , probs )
    print(f"✔  {tag}: 生成 counts/probs  (targets={counts.shape[0]-1}, values={counts.shape[1]-1})")

def main() -> None:
    done = set()
    # 掃描所有 npz / shard 拿到 tag
    for p in itertools.chain(DATA_DIR.glob("*_memory_part*.npz"),
                             DATA_DIR.glob("*_targets_p*.npy")):
        m = PAT_NPZ.match(p.name) or PAT_SHARD.match(p.name)
        if not m:
            continue
        tag = m.group("tag")
        if tag not in done:
            convert_one(tag)
            done.add(tag)

    if not done:
        print("⚠️  找不到任何適用檔案")
    else:
        print("✅  全部統計完成")

if __name__ == "__main__":
    main()
