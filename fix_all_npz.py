#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 samples/ 與 out_npz/ 目錄下的 *.npz 統一轉成 (N, R, C) 之三維陣列。
N = R * C  (1..R*C 對應刮刮樂號碼)

使用方式
--------
$ python fix_all_npz.py                      # 實際寫檔
$ python fix_all_npz.py --dirs dir1 dir2     # 自訂多個目錄
$ python fix_all_npz.py --dry-run            # 只列出將採取的行動

作者：ChatGPT
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────


def parse_size(fname: str) -> Tuple[int, int] | None:
    """從檔名擷取 rows × cols，例如 4x5、10x12 …"""
    m = re.search(r"(\d+)x(\d+)", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def standardize_array(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    把輸入 array 轉成 (rows*cols, rows, cols)。
    規則：
    1. (rows, cols)  → replicate 成 N×R×C
    2. (rows, cols, N) → swapaxes(0, 2)
    3. 其他 3D → 嘗試把其中兩軸是 rows/cols 的情況 reorder
    """
    if arr.ndim == 2:  # 單張平面 => 複製成每個號碼同一張
        n = rows * cols
        return np.broadcast_to(arr, (n, rows, cols)).copy()

    if arr.ndim == 3:
        # 先猜 shape = (N, R, C)
        if arr.shape == (rows * cols, rows, cols):
            return arr
        # 接著猜 shape = (R, C, N)
        if arr.shape == (rows, cols, rows * cols):
            return np.transpose(arr, (2, 0, 1))
        # 最後嘗試把哪兩軸剛好是 rows, cols 找出來
        axes = list(arr.shape)
        try:
            r_idx = axes.index(rows)
            c_idx = axes.index(cols)
        except ValueError:
            raise ValueError("無法判斷哪兩軸是 rows/cols") from None
        n_idx = 3 - r_idx - c_idx
        reordered = np.moveaxis(arr, (n_idx, r_idx, c_idx), (0, 1, 2))
        if reordered.shape[1:] == (rows, cols):
            return reordered

    raise ValueError(f"無法轉換 shape {arr.shape} -> ({rows}x{cols})")


def process_file(path: Path, dry_run: bool = False) -> None:
    size = parse_size(path.name)
    if not size:
        print(f"  ↳ 跳過（無法解析尺寸）：{path.name}")
        return
    rows, cols = size
    data = np.load(path)
    key = data.files[0] if data.files else None
    if key is None:
        print(f"  ↳ 跳過（空 NPZ）：{path.name}")
        return
    arr = data[key]
    try:
        fixed = standardize_array(arr, rows, cols)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  ✗  {path.name}: 解析失敗 → {exc}")
        return

    if arr.shape == fixed.shape:
        print(f"  ✓  {path.name}: 形狀 OK {arr.shape}")
        return

    print(
        f"  ➜  {path.name}: {arr.shape}  →  {fixed.shape}"
        f"{'   (dry-run)' if dry_run else ''}"
    )
    if not dry_run:
        # 直接覆寫原檔，保持原來的 key 名稱
        np.savez_compressed(path, **{key: fixed})


def main() -> None:
    parser = argparse.ArgumentParser(description="修復 *.npz 維度")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["samples", "out_npz"],
        help="要掃描的資料夾（預設 samples/ out_npz/）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只顯示將進行的動作，不修改檔案",
    )
    args = parser.parse_args()

    for d in args.dirs:
        dir_path = Path(d)
        if not dir_path.is_dir():
            print(f"資料夾不存在：{dir_path}", file=sys.stderr)
            continue
        print(f"\n=== 處理資料夾：{dir_path} ===")
        for npz in sorted(dir_path.glob("*.npz")):
            process_file(npz, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
