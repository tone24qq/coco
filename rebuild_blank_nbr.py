#!/usr/bin/env python3
"""
rebuild_blank_nbr.py
------------------------------------------------------------
for every out_npz/full_stats_{rows}x{cols}.npz
    → recompute `nbr3x3_blank` (blank-center 3×3 neighbour counts)
    → overwrite the array in-place (mmap r+)

⚠️ 只改這張陣列，freq / row_freq … 完全保留
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ───── 使用者參數 ──────────────────────────────────────────
NPZ_DIR = Path("samples")  # 統計檔所在資料夾
SAMPLES_DIR = Path("samples")  # ZIP / JSON 根目錄
BLANK_VAL = -1  # 你的空格值；若用 0 改成 0
SHOW_SKIP = True  # 跳過壞盤時是否列印警告
# ─────────────────────────────────────────────────────────


def iter_boards(rows: int, cols: int):
    """yield 每一盤合法 (rows×cols) board (np.ndarray)"""
    for zp_path in SAMPLES_DIR.rglob("*.zip"):
        with zipfile.ZipFile(zp_path) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".json"):
                    continue

                # ① 先用檔名 rowsxcols 快速過濾
                stem = Path(name).stem
                try:
                    r, c = map(int, stem.split("x"))
                except ValueError:
                    r = c = None
                if (r, c) != (rows, cols):
                    # 可能是 dict 格式，得讀檢查
                    data = json.loads(zf.read(name))
                    if isinstance(data, dict):
                        if (data.get("rows"), data.get("cols")) != (rows, cols):
                            continue
                        boards = [data["grid"]]
                    else:
                        continue
                else:
                    data = json.loads(zf.read(name))
                    boards = (
                        data
                        if (
                            isinstance(data, list)
                            and data
                            and isinstance(data[0], list)
                            and isinstance(data[0][0], list)
                        )
                        else [data]
                    )

                for b in boards:
                    arr = np.asarray(b, dtype=np.int16)
                    if arr.ndim != 2 or arr.shape != (rows, cols):
                        if SHOW_SKIP:
                            tqdm.write(
                                f"⚠️ skip irregular board in {name}: shape={arr.shape}"
                            )
                        continue
                    yield arr


def build_blank_nbr(npz_path: Path):
    rows, cols = map(int, npz_path.stem.split("_")[-1].split("x"))
    V = rows * cols + 1
    nbr = np.zeros((V, 3, 3), dtype=np.uint32)

    for arr in tqdm(iter_boards(rows, cols), desc=npz_path.name, unit="board"):
        pad = np.pad(arr, 1, constant_values=BLANK_VAL)
        for r in range(rows):
            for c in range(cols):
                if arr[r, c] != BLANK_VAL:
                    continue
                k = pad[r : r + 3, c : c + 3]
                for dy in range(3):
                    for dx in range(3):
                        nbr[k[dy, dx], dy, dx] += 1

    # —— 寫回 npz (就地覆蓋) ―――――――――――――――――――――――――――――――
    with np.load(npz_path, mmap_mode="r+") as z:
        if "nbr3x3_blank" not in z.files or z["nbr3x3_blank"].shape != nbr.shape:
            raise ValueError(
                f"{npz_path.name} 內的 nbr3x3_blank shape 不符，請先確認檔案版本"
            )
        z["nbr3x3_blank"][:] = nbr
    print(f"✅ {npz_path.name} done ─ samples={int(nbr.sum()) // 9:,}  max={nbr.max()}")


def main():
    npz_files = sorted(NPZ_DIR.glob("full_stats_*x*.npz"))
    if not npz_files:
        sys.exit(f"❌ {NPZ_DIR.resolve()} 找不到 full_stats_*.npz")

    for f in npz_files:
        build_blank_nbr(f)


if __name__ == "__main__":
    main()
