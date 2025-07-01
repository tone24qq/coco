import logging
from pathlib import Path

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
