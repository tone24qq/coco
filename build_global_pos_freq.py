#!/usr/bin/env python
"""
把現有樣本（npz / zip / json / jsonl）統計成
    out_npz/global_pos_freq_<rows>x<cols>.npz

用法：
    python build_global_pos_freq.py -s samples -o out_npz

Author : 橘子的小助手 (2025-07-11)
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import zipfile
from collections import defaultdict
from typing import Iterable, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# 可選進度條（tqdm 不存在就用假進度）
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it: Iterable, **kwargs):  # type: ignore
        return it
# -----------------------------------------------------------------------------
# 可選高速 JSON（orjson 不存在就退回標準 json）
try:
    import orjson as _json
except ImportError:  # pragma: no cover
    _json = None
# -----------------------------------------------------------------------------

BLANK_VAL = -1  # 盤面遮蔽標記


def _loads(buf: bytes):
    """兼容 orjson / 標準 json."""
    if _json:
        return _json.loads(buf)
    return json.loads(buf.decode())


# ------------------------------------------------------------------ #
# 1. 掃描樣本檔，Yield (rows, cols, board ndarray)                   #
# ------------------------------------------------------------------ #
def iter_boards(samples_dir: pathlib.Path) -> Iterable[Tuple[int, int, np.ndarray]]:
    """遍歷資料夾，支援 .npz / .zip / .json / .jsonl."""
    for p in samples_dir.rglob("*"):
        suffix = p.suffix.lower()
        if suffix == ".npz":  # ---- 直接載 ndarray ----
            with np.load(p) as data:
                if "boards" not in data:
                    continue
                arr = data["boards"].astype(int)
            if arr.ndim == 2:  # 單張 → 升維
                arr = arr[None, ...]
            for bd in arr:
                yield bd.shape[0], bd.shape[1], bd

        elif suffix == ".zip":  # ---- zip 裡面放 json ----
            with zipfile.ZipFile(p) as zf:
                for name in zf.namelist():
                    if not name.endswith(".json"):
                        continue
                    bd = np.array(_loads(zf.read(name)), dtype=int)
                    yield bd.shape[0], bd.shape[1], bd

        elif suffix in {".json", ".jsonl"}:  # ---- 單檔或行分隔 ----
            with p.open("rb") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    bd = np.array(_loads(line), dtype=int)
                    yield bd.shape[0], bd.shape[1], bd


# ------------------------------------------------------------------ #
# 2. 累積 counts[r, c, num] → 機率 freq                              #
# ------------------------------------------------------------------ #
def build_freq(samples_dir: str):
    freq_dict: dict[Tuple[int, int], np.ndarray] = defaultdict(lambda: None)  # type: ignore
    for r, c, bd in tqdm(iter_boards(pathlib.Path(samples_dir)),
                         desc="Scanning sample boards"):
        max_num = r * c
        counts = freq_dict.get((r, c))
        if counts is None:
            counts = np.zeros((r, c, max_num + 1), dtype=np.int64)
            freq_dict[(r, c)] = counts

        for i, j in itertools.product(range(r), range(c)):
            val = int(bd[i, j])
            if val == BLANK_VAL:
                continue
            counts[i, j, val] += 1

    # 轉機率
    for shape, cnt in freq_dict.items():
        totals = cnt.sum(axis=2, keepdims=True)
        totals[totals == 0] = 1
        freq_dict[shape] = cnt / totals
    return freq_dict


# ------------------------------------------------------------------ #
# 3. 存成 global_pos_freq_*                                          #
# ------------------------------------------------------------------ #
def dump_freq(freq_dict, output_dir: str):
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for (r, c), freq in freq_dict.items():
        path = out_dir / f"global_pos_freq_{r}x{c}.npz"
        np.savez_compressed(path, freq=freq.astype(np.float32))
        mb = path.stat().st_size / 1024 / 1024
        print(f"✅ 產生 {path.name:28s}  shape={freq.shape}  ({mb:.1f} MB)")


# ------------------------------------------------------------------ #
# 4. CLI 入口                                                        #
# ------------------------------------------------------------------ #
def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="build_global_pos_freq",
        description="從樣本檔產生 global_pos_freq_<shape>.npz"
    )
    parser.add_argument(
        "-s", "--samples", required=True,
        help="樣本資料夾 (支援 npz/zip/json/jsonl 混放)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="輸出 out_npz 目錄"
    )
    args = parser.parse_args(argv)
    freq_dict = build_freq(args.samples)
    if not freq_dict:
        print("❌ 沒掃到任何板子！確認 --samples 路徑正確？", file=sys.stderr)
        sys.exit(1)
    dump_freq(freq_dict, args.output)
    print("🎉 全部完成！")


if __name__ == "__main__":
    main()
