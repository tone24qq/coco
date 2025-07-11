#!/usr/bin/env python3
# 將 samples/pos_freq_<rows>x<cols>.npz 壓成 (rows, cols) 並正規化
import argparse
from pathlib import Path
import numpy as np
import re

PAT = re.compile(r"pos_freq_(\d+)x(\d+)\.npz$", re.I)

def fix_file(path: Path, backup: bool = False):
    dat = np.load(path)
    key = dat.files[0] if dat.files else None
    if not key:
        print(f"[SKIP] {path.name} 無有效 array")
        return
    arr = dat[key]
    changed = False

    # 壓縮到 2-D
    if arr.ndim > 2:
        arr = arr.sum(axis=tuple(range(2, arr.ndim)))
        changed = True

    # 修負值 & 正規化
    arr[arr < 0] = 0.0
    total = arr.sum()
    if not np.isclose(total, 1.0):
        arr = arr / (total or 1.0)
        changed = True

    if changed:
        if backup:
            path.rename(path.with_suffix(".npz.bak"))
        np.savez(path, arr)
        print(f"[FIX ] {path.name} -> shape={arr.shape}, sum=1.0")
    else:
        print(f"[OK  ] {path.name}")

def main():
    ap = argparse.ArgumentParser(description="修正 samples/pos_freq_*.npz 三維檔")
    ap.add_argument("--dir", default="samples", help="samples 目錄")
    ap.add_argument("--backup", action="store_true", help="備份 .bak")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"資料夾不存在：{root}")
        return

    n = 0
    for f in root.glob("pos_freq_*.npz"):
        if PAT.search(f.name):
            fix_file(f, backup=args.backup)
            n += 1
    print(f"處理完畢，共 {n} 個檔案")

if __name__ == "__main__":
    main()
