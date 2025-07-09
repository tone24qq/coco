import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from analyzer import iter_sample_jsons

logger = logging.getLogger(__name__)


def build_position_prior(samples_dir: str, outfile: str, buckets: int = 20) -> None:
    """Build global position prior from ``samples_dir`` and save to ``outfile``."""
    path = Path(samples_dir)
    # first pass to determine max digit
    max_digit = 0
    for sample in iter_sample_jsons(str(path)):
        grid = np.asarray(sample["grid"], dtype=int)
        if grid.size:
            max_digit = max(max_digit, int(grid.max()))
    if max_digit == 0:
        raise ValueError("no valid samples found")

    counts = np.zeros((max_digit + 1, buckets, buckets), dtype=np.int64)
    for sample in iter_sample_jsons(str(path)):
        grid = np.asarray(sample["grid"], dtype=int)
        rows, cols = sample["rows"], sample["cols"]
        for r in range(rows):
            for c in range(cols):
                k = int(grid[r, c])
                if k <= 0:
                    continue
                u = r / (rows - 1) if rows > 1 else 0.0
                v = c / (cols - 1) if cols > 1 else 0.0
                i = min(int(u * buckets), buckets - 1)
                j = min(int(v * buckets), buckets - 1)
                counts[k, i, j] += 1

    totals = counts.sum(axis=(1, 2), keepdims=True)
    totals[totals == 0] = 1
    freq = counts.astype(float) / totals
    np.savez_compressed(outfile, freq=freq)
    logger.info("position prior saved to %s", outfile)
    # 中文說明：完成全域位置先驗計算並寫入檔案


def build_position_prior_map(
    samples_dir: str, buckets: int = 20
) -> Dict[Tuple[int, int], Dict[Tuple[int, int], Dict[int, float]]]:
    """Return per-board position probability maps for all sample sizes."""
    from analyzer import compute_position_probabilities

    dims: set[Tuple[int, int]] = set()
    for sample in iter_sample_jsons(str(samples_dir)):
        dims.add((sample["rows"], sample["cols"]))

    priors: Dict[Tuple[int, int], Dict[Tuple[int, int], Dict[int, float]]] = {}
    for rows, cols in sorted(dims):
        priors[(rows, cols)] = compute_position_probabilities(samples_dir, rows, cols)
    logger.info("position prior map built for %d board sizes", len(priors))
    # 中文說明：產生的先驗表涵蓋的不同盤面數量
    return priors
