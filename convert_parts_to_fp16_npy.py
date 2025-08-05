#!/usr/bin/env python
"""convert_parts_to_fp16_npy.py  ── 把 *_memory_part*.npz → float16 .npy

用法：
    1. 把此檔存到 repo/tools 下。
    2. cd 到專案根目錄執行 `python tools/convert_parts_to_fp16_npy.py`
    3. 轉檔成功後，確認新檔大小 OK，再視需要刪掉舊 .npz
"""

from __future__ import annotations
import re, glob
from pathlib import Path
import numpy as np

SRC_DIR = Path("data_archives")               # 如有不同路徑自行修改
PAT  = re.compile(r"(\d+)x(\d+)_memory_part(\d+)\.npz")

shapes: dict[tuple[int, int], list[tuple[int, Path]]] = {}

# 1️⃣ 收集所有 part
for p in SRC_DIR.glob("*_memory_part*.npz"):
    m = PAT.match(p.name)
    if not m:
        continue
    r, c, part_idx = map(int, m.groups())
    shapes.setdefault((r, c), []).append((part_idx, p))

# 2️⃣ 逐 shape 合併、轉 dtype、寫 .npy
for (rows, cols), lst in shapes.items():
    lst.sort()                                               # part0, part1, …
    keys_lst, vals_lst = [], []

    for _, path in lst:
        with np.load(path) as z:
            keys_lst.append(z["keys"])
            vals_lst.append(z["values"])

    keys = np.concatenate(keys_lst, axis=0).astype(np.float16, copy=False)
    vals = np.concatenate(vals_lst, axis=0).astype(np.float16, copy=False)

    k_path = SRC_DIR / f"{rows}x{cols}_keys.npy"
    v_path = SRC_DIR / f"{rows}x{cols}_vals.npy"

    # 用 memmap 寫檔，避免一次佔用太大 RAM
    np.lib.format.open_memmap(k_path, mode="w+", dtype=keys.dtype,
                              shape=keys.shape)[:] = keys
    np.lib.format.open_memmap(v_path, mode="w+", dtype=vals.dtype,
                              shape=vals.shape)[:] = vals

    print(f"✅ {rows}x{cols}: parts={len(lst)} → {keys.shape[0]} samples | "
          f"keys={k_path.stat().st_size/1024/1024:.1f} MB")

print("\n🎉 轉檔完成！請確認檔案大小後，再視需要刪除舊 .npz")