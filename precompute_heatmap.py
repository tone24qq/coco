"""Precompute heatmap and cell frequency statistics from datasets."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm

from dataset import BLANK_VALUE
from utils.io_utils import _extract_boards


def _update_stats(
    board: np.ndarray,
    target: int,
    heatmaps: Dict[Tuple[int, int], np.ndarray],
    counts: Dict[Tuple[int, int], np.ndarray],
) -> None:
    rows, cols = board.shape
    shape = (rows, cols)
    if shape not in heatmaps:
        heatmaps[shape] = np.ones((rows, cols), dtype=np.float64)
        counts[shape] = np.ones((rows, cols), dtype=np.float64)
    pos = np.argwhere(board == target)
    if pos.size > 0:
        r, c = pos[0]
        heatmaps[shape][r, c] += 1
    counts[shape] += (board != BLANK_VALUE).astype(np.float64)


def collect_statistics(
    data_dir: str,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """Traverse ``data_dir`` and return heatmap and count statistics."""
    heatmaps: Dict[Tuple[int, int], np.ndarray] = {}
    counts: Dict[Tuple[int, int], np.ndarray] = {}
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
    heatmaps: Dict[Tuple[int, int], np.ndarray],
    counts: Dict[Tuple[int, int], np.ndarray],
    out_dir: str,
) -> None:
    """Save statistics to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for (rows, cols), mat in heatmaps.items():
        arr = mat / mat.sum()
        np.save(out / f"heatmap_{rows}x{cols}.npy", arr)
    for (rows, cols), mat in counts.items():
        arr = mat / mat.sum()
        np.save(out / f"counts_{rows}x{cols}.npy", arr)


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute heatmap priors")
    parser.add_argument("--data_dir", default="data_archives")
    parser.add_argument("--out_dir", default="priors")
    ns = parser.parse_args(args)
    heat, cnt = collect_statistics(ns.data_dir)
    save_statistics(heat, cnt, ns.out_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
