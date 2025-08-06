#!/usr/bin/env python3
"""
把舊的  {rows}x{cols}_memory_part?.npz  轉成
    {rows}x{cols}_keys_p{i}.npy     (float16)
    {rows}x{cols}_vals_p{i}.npy     (float16)
    {rows}x{cols}_targets_p{i}.npy  (int16)
    {rows}x{cols}_boards_p{i}.npy   (int8，可選)

* 每個輸出檔大小 ≤ MAX_MB  (預設 100 MB)
* 轉完後 **可選** 刪掉舊 npz，只要解除最後幾行註解即可。
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import re, math, itertools, typing as _t

# ---------------- 可自行調整 ---------------- #
DATA_DIR = Path("data_archives")   # 舊檔所在，也會把新檔寫進去
MAX_MB   = 100                     # 單檔上限 (MiB 約略值)
# ------------------------------------------ #

_PART_RE = re.compile(r"(?P<tag>\d+x\d+)_memory_part\d+\.npz$")

def _gather(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, _t.Optional[np.ndarray]]:
    """讀取同一尺寸的所有 part，做 concat & 降精度。"""
    keys_l, vals_l, tgt_l, brd_l = [], [], [], []
    for p in sorted(DATA_DIR.glob(f"{tag}_memory_part*.npz")):
        with np.load(p) as z:
            keys_l.append(z["keys"].astype(np.float16, copy=False))
            vals_l.append(z["values"].astype(np.float16, copy=False))
            tgt_l.append( z["targets"].astype(np.int16,  copy=False))
            if "boards" in z.files:
                brd_l.append(z["boards"].astype(np.int8, copy=False))

    keys    = np.concatenate(keys_l)
    vals    = np.concatenate(vals_l)
    targets = np.concatenate(tgt_l)
    boards  = np.concatenate(brd_l) if brd_l else None
    return keys, vals, targets, boards

def _rows_per_shard(keys: np.ndarray, vals: np.ndarray, boards: _t.Optional[np.ndarray]) -> int:
    bytes_per_row = (keys.shape[1] + vals.shape[1]) * 2 + 2  # float16=2 bytes, int16=2
    if boards is not None:
        bytes_per_row += boards.shape[1]                      # int8 = 1 byte
    return max(1, math.floor(MAX_MB * 1_000_000 / bytes_per_row))

def _shard_save(arr: np.ndarray, base: str, rows_per: int, dtype_note: str) -> None:
    for i, start in enumerate(range(0, len(arr), rows_per)):
        shard = arr[start:start + rows_per]
        np.save(DATA_DIR / f"{base}_p{i}.npy", shard)
        print(f"  · {base}_p{i}.npy  {shard.shape}  ({dtype_note})")

def convert_one(tag: str) -> None:
    print(f"▶  轉換 {tag}")
    keys, vals, targets, boards = _gather(tag)
    rps = _rows_per_shard(keys, vals, boards)

    _shard_save(keys,    f"{tag}_keys",    rps, "f16")
    _shard_save(vals,    f"{tag}_vals",    rps, "f16")
    _shard_save(targets, f"{tag}_targets", rps, "i16")
    if boards is not None:
        _shard_save(boards, f"{tag}_boards", rps, "i8")

def main() -> None:
    done = set()
    for npz in sorted(DATA_DIR.glob("*_memory_part*.npz")):
        m = _PART_RE.match(npz.name)
        if not m:
            continue
        tag = m.group("tag")
        if tag not in done:
            convert_one(tag)
            done.add(tag)
            # 若要立刻刪舊檔以省空間，取消下行註解
            # for old in DATA_DIR.glob(f"{tag}_memory_part*.npz"): old.unlink()

    if not done:
        print("⚠️  找不到任何 *_memory_part*.npz 檔")
    else:
        print("✅  全部轉換完成")

if __name__ == "__main__":
    main()