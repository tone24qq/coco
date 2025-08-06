#!/usr/bin/env python3
"""
將舊的 {rows}x{cols}_memory_part?.npz 轉成
  {rows}x{cols}_keys_p{i}.npy     float16  (≤100 MB)
  {rows}x{cols}_vals_p{i}.npy     float16
  {rows}x{cols}_targets_p{i}.npy  int16
  {rows}x{cols}_boards_p{i}.npy   int8   (若舊檔含 boards)
─  任何單檔大小都不會超過 MAX_MB。
"""

from __future__ import annotations
from pathlib import Path
import numpy as np, re, math, json, sys

DATA_DIR = Path("data_archives")     # 舊檔／新檔皆放在此
MAX_MB   = 100                       # 單檔 ≤ 100 MB
F16      = 2                         # float16 / int16 byte 數
PAT      = re.compile(r"(\d+)x(\d+)_memory_part\d+\.npz")

def gather_parts(shape_tag: str):
    keys_lst, vals_lst, tgt_lst, brd_lst = [], [], [], []
    for p in sorted(DATA_DIR.glob(f"{shape_tag}_memory_part*.npz")):
        with np.load(p) as z:
            keys_lst.append(z["keys"].astype(np.float16, copy=False))
            vals_lst.append(z["values"].astype(np.float16, copy=False))
            tgt_lst.append(z["targets"].astype(np.int16,  copy=False))
            if "boards" in z.files:                       # boards 可能不存在
                brd_lst.append(z["boards"].astype(np.int8, copy=False))
    keys    = np.concatenate(keys_lst)
    vals    = np.concatenate(vals_lst)
    targets = np.concatenate(tgt_lst)
    boards  = np.concatenate(brd_lst) if brd_lst else None
    return keys, vals, targets, boards

def shard_and_save(arr: np.ndarray, base: str, rows_per_shard: int, dtype_hint:str):
    for i, s in enumerate(range(0, len(arr), rows_per_shard)):
        e = s + rows_per_shard
        np.save(DATA_DIR / f"{base}_p{i}.npy", arr[s:e])
        print(f"  · {base}_p{i}.npy  ({dtype_hint})  rows {s}-{e-1}")

def main() -> None:
    shapes_done = set()
    for part in sorted(DATA_DIR.glob("*_memory_part*.npz")):
        m = PAT.match(part.name)
        if not m: continue
        rows, cols = map(int, m.groups())
        tag = f"{rows}x{cols}"
        if tag in shapes_done:    # 避免重複轉
            continue
        shapes_done.add(tag)
        print(f"▶  轉檔 {tag}")

        keys, vals, targets, boards = gather_parts(tag)

        # 計算每列 byte 數，依 MAX_MB 推出 rows_per_shard
        bytes_per_row = (keys.shape[1] + vals.shape[1]) * F16 + F16
        if boards is not None:
            bytes_per_row += boards.shape[1]      # boards 存 int8, 1 byte
        rows_per_shard = max(1, math.floor(MAX_MB*1_000_000 / bytes_per_row))

        shard_and_save(keys,    f"{tag}_keys",    rows_per_shard, "float16")
        shard_and_save(vals,    f"{tag}_vals",    rows_per_shard, "float16")
        shard_and_save(targets, f"{tag}_targets", rows_per_shard, "int16")
        if boards is not None:
            shard_and_save(boards, f"{tag}_boards", rows_per_shard, "int8")

        # 如要刪舊檔，解除註解：
        # for p in DATA_DIR.glob(f"{tag}_memory_part*.npz"):
        #     p.unlink()

    print("✅  轉檔完成")

if __name__ == "__main__":
    main()