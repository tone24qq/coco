

"""Precompute heatmap and cell frequency statistics from datasets."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm

from dataset import BLANK_VALUE
from utils.io_utils import _extract_boards

# ---------- 參數可按需調整 ----------
ALPHA: float = 1e-3  # Dirichlet 平滑強度
CONSIDER_BLANK_ONLY = True  # True ➔ 只把「還沒翻的格子」算進分母


class Bucket(Enum):
    SMALL = "small"  #  1–5
    MID = "mid"  #  6–15
    LARGE = "large"  # 16–20


def bucket_of(val: int) -> Bucket:
    if val <= 5:
        return Bucket.SMALL
    if val <= 15:
        return Bucket.MID
    return Bucket.LARGE


def _update_stats(
    board: np.ndarray,
    target: int,
    heatmaps: Dict[Tuple[int, int, Bucket], np.ndarray],
    counts: Dict[Tuple[int, int, Bucket], np.ndarray],
) -> None:
    rows, cols = board.shape
    shape = (rows, cols, bucket_of(target))
    if shape not in heatmaps:
        heatmaps[shape] = np.zeros((rows, cols), dtype=np.float64)
        counts[shape] = np.zeros((rows, cols), dtype=np.float64)

    # 分子：該格真的放了目標值
    pos = np.argwhere(board == target)
    if pos.size > 0:
        r, c = pos[0]
        heatmaps[shape][r, c] += 1

    # 分母：此格「可能被選中」的次數
    if CONSIDER_BLANK_ONLY:
        counts[shape] += (board == BLANK_VALUE).astype(np.float64)
    else:
        counts[shape] += 1.0  # 把每局都算一次


def collect_statistics(
    data_dir: str,
) -> Tuple[
    Dict[Tuple[int, int, Bucket], np.ndarray],
    Dict[Tuple[int, int, Bucket], np.ndarray],
]:
    """Traverse ``data_dir`` and return heatmap and count statistics."""
    heatmaps: Dict[Tuple[int, int, Bucket], np.ndarray] = {}
    counts: Dict[Tuple[int, int, Bucket], np.ndarray] = {}
    files: list[Path] = []
    for root, _, fns in os.walk(data_dir):
        for name in fns:
            if name.endswith(".json") or name.endswith(".zip"):
                files.append(Path(root) / name)
    for path in tqdm(files, desc="processing files"):
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for inner in zf.namelist():
                    if inner.endswith(".json"):
                        with zf.open(inner) as f:
                            obj = json.load(f)
                        for board, target in _extract_boards(obj):
                            _update_stats(board, target, heatmaps, counts)
        else:
            with open(path) as f:
                obj = json.load(f)
            for board, target in _extract_boards(obj):
                _update_stats(board, target, heatmaps, counts)
    return heatmaps, counts


def save_statistics(
    heatmaps: Dict[Tuple[int, int, Bucket], np.ndarray],
    counts: Dict[Tuple[int, int, Bucket], np.ndarray],
    out_dir: str,
) -> None:
    """Save statistics to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (rows, cols, bucket), hits in heatmaps.items():
        denom = counts[(rows, cols, bucket)]
        # Dirichlet 平滑，避免 0 / 0
        prior = (hits + ALPHA) / (denom + ALPHA * (rows * cols))
        fname = f"heatmap_{bucket.value}_{rows}x{cols}.npy"
        np.save(out / fname, prior)


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute heatmap priors")
    parser.add_argument("--data_dir", default="data_archives")
    parser.add_argument("--out_dir", default="priors")
    ns = parser.parse_args(args)
    heat, cnt = collect_statistics(ns.data_dir)
    save_statistics(heat, cnt, ns.out_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
