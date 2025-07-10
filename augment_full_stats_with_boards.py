#!/usr/bin/env python3
"""
將 samples/full_stats_*x*.npz 的 boards 拆成 <100 MB 小檔 (uint8)。
輸出檔名: boards_{rows}x{cols}_part{N}.npz
執行: python trim_boards_to_100mb.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import List

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **k):
        return it


# 路徑與檔名規則 -------------------------------------------------
SAMPLES_DIR = Path("samples")
NPZ_RE = re.compile(r"full_stats_(\d+)x(\d+)\.npz$")
OUT_TMPL = "boards_{rows}x{cols}_part{idx}.npz"
MAX_FILE_MB = 100  # 每檔上限 (MB)
DTYPE = np.uint8  # 1 byte


# ---------------------------------------------------------------
def to_int(x, default=-999):
    try:
        return int(str(x).strip().split(".")[0])
    except Exception:
        return default


def collect_boards(rows: int, cols: int) -> np.ndarray | None:
    """從 ZIP 取出符合尺寸的盤面。回傳 uint8 ndarray 或 None。"""
    boards: List[np.ndarray] = []
    for zp in SAMPLES_DIR.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                for name in zf.namelist():
                    if not name.endswith(".json"):  # 非 JSON
                        continue
                    try:
                        data = json.loads(zf.read(name))
                    except json.JSONDecodeError:
                        continue

                    # list-JSON
                    if isinstance(data, list):
                        for b in data:
                            if (
                                isinstance(b, list)
                                and len(b) == rows
                                and all(len(r) == cols for r in b)
                            ):
                                boards.append(np.asarray(b, DTYPE))
                        continue

                    # dict-JSON
                    if (
                        isinstance(data, dict)
                        and to_int(data.get("rows")) == rows
                        and to_int(data.get("cols")) == cols
                        and isinstance(data.get("grid"), list)
                    ):
                        g = data["grid"]
                        if len(g) == rows and all(len(r) == cols for r in g):
                            boards.append(np.asarray(g, DTYPE))
        except zipfile.BadZipFile:
            continue
    if not boards:
        return None
    return np.stack(boards)  # (N, rows, cols), dtype=uint8


def main() -> None:
    npz_files = sorted(SAMPLES_DIR.rglob("full_stats_*x*.npz"))
    if not npz_files:
        print("❌ 找不到 full_stats_*x*.npz，確認路徑。")
        sys.exit(1)

    for stats_npz in tqdm(npz_files, desc="處理尺寸檔", unit="file"):
        m = NPZ_RE.search(stats_npz.name)
        if not m:
            tqdm.write(f"⚠️ 無法解析尺寸：{stats_npz.name}")
            continue
        rows, cols = map(int, m.groups())

        # 若已存在 boards_..._part0.npz 就略過
        if list(SAMPLES_DIR.glob(OUT_TMPL.format(rows=rows, cols=cols, idx="*"))):
            tqdm.write(f"🟢 已有 boards_{rows}x{cols}_part*.npz，略過")
            continue

        tqdm.write(f"➜ 收集 {rows}x{cols} boards…")
        arr = collect_boards(rows, cols)
        if arr is None:
            tqdm.write(f"⚠️ 找不到任何 {rows}x{cols} 盤面")
            continue

        # 估算每盤 bytes，決定切檔大小
        bytes_per_board = rows * cols * DTYPE().nbytes
        max_boards_per_file = max(1, (MAX_FILE_MB * 1024 * 1024) // bytes_per_board)
        parts = math.ceil(arr.shape[0] / max_boards_per_file)

        for idx in range(parts):
            sl = slice(idx * max_boards_per_file, (idx + 1) * max_boards_per_file)
            part_arr = arr[sl]
            out_path = SAMPLES_DIR / OUT_TMPL.format(rows=rows, cols=cols, idx=idx)
            np.savez_compressed(out_path, boards=part_arr)
            sz = out_path.stat().st_size / 1024 / 1024
            tqdm.write(
                f"  ✅ {out_path.name}  {part_arr.shape[0]} boards → {sz:.2f} MB"
            )

    print("🎉 全部尺寸補檔完成 (單檔 ≤100 MB)")


if __name__ == "__main__":
    main()
