#!/usr/bin/env python3
"""
從 {rows}x{cols}.json 取出 board，產生
    {rows}x{cols}_boards_p{i}.npy  (int8)

會自動對齊目前 keys/vals/targets 的分片行數，保持 part 對應。
"""

from __future__ import annotations
from pathlib import Path
import json, re, math
import numpy as np

# ---------------- 可調參數 ---------------- #
DATA_DIR = Path("data_archives")   # 存放 .json 與 .npy 的資料夾
MAX_MB   = 100                     # 若沒有既有分片，預設單檔 ≤ MAX_MB
# ---------------------------------------- #

_PART_RE = re.compile(r"(?P<tag>\d+x\d+)_keys_p(?P<idx>\d+)\.npy$")

def rows_per_existing_part(tag: str) -> list[int] | None:
    """讀出現有 keys/vals/targets 每一分片的行數；若沒有則回傳 None。"""
    parts = []
    for p in sorted(DATA_DIR.glob(f"{tag}_keys_p*.npy"),
                    key=lambda x: int(_PART_RE.search(x.name).group("idx"))):
        parts.append(np.load(p, mmap_mode="r").shape[0])
    return parts or None

def split_and_save(arr: np.ndarray, tag: str, part_rows: list[int]) -> None:
    start = 0
    for i, rows in enumerate(part_rows):
        shard = arr[start:start + rows]
        out = DATA_DIR / f"{tag}_boards_p{i}.npy"
        np.save(out, shard.astype(np.int8, copy=False))
        print(f"✅  saved {out.name:20s}  {shard.shape}")
        start += rows
    assert start == len(arr), "row count mismatch when slicing boards"

def calc_rows_per_shard(example_row_bytes: int, total_rows: int) -> list[int]:
    per = max(1, math.floor(MAX_MB * 1_000_000 / example_row_bytes))
    parts = []
    i = 0
    while i < total_rows:
        parts.append(min(per, total_rows - i))
        i += per
    return parts

def process_json(json_path: Path) -> None:
    tag = json_path.stem                              # e.g. "4x5"
    rows, cols = map(int, tag.split("x"))
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # 把 board list 轉為 (N, rows*cols) int8
    boards = np.empty((len(data), rows * cols), dtype=np.int8)
    for i, item in enumerate(data):
        board = np.asarray(item["board"], dtype=np.int8).reshape(rows * cols)
        boards[i] = board
    print(f"▶  {tag}:  loaded {len(boards)} boards from JSON")

    # 取得既有分片大小；若沒有則計算平均
    part_rows = rows_per_existing_part(tag)
    if part_rows is None:
        bytes_per_row = rows * cols                  # int8 → 1 byte
        part_rows = calc_rows_per_shard(bytes_per_row, len(boards))
        print(f"   (no existing shards, auto chunk: {part_rows})")

    split_and_save(boards, tag, part_rows)

def main() -> None:
    json_files = list(DATA_DIR.glob("*x*.json"))
    if not json_files:
        print("⚠️  找不到任何 *.json，結束")
        return
    for jp in json_files:
        try:
            process_json(jp)
        except Exception as e:
            print(f"❌  轉換 {jp.name} 失敗：{e}")

if __name__ == "__main__":
    main()