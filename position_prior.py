import logging

import numpy as np

from analyzer import sample_excel_boards

logger = logging.getLogger(__name__)


def build_position_prior(
    rows: int,
    cols: int,
    outfile: str,
    buckets: int = 20,
    *,
    n_synth: int = 1000,
    seed: int = 0
) -> None:
    """Build global position prior from synthetic boards and save to ``outfile``."""
    boards = sample_excel_boards(rows, cols, n_synth, seed)
    max_digit = rows * cols
    counts = np.zeros((max_digit + 1, buckets, buckets), dtype=np.int64)
    for board in boards:
        for r in range(rows):
            for c in range(cols):
                k = int(board[r, c])
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


def load_position_prior(path: str) -> np.ndarray:
    """Load position prior tensor from ``path``."""
    arr = np.load(path)["freq"]
    return np.asarray(arr, dtype=float)
