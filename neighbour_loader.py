import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

NBR_DIR = Path(os.environ.get("NBR_DATA_DIR", "data_archives"))
NEIGHBOR_PROBS: Dict[Tuple[int, int], np.ndarray] = {}


def load_nbr(rows: int, cols: int) -> None:
    """Load neighbour probability matrix for a given board shape.

    Parameters
    ----------
    rows, cols:
        Shape of the board; used to construct the filename
        ``"{rows}x{cols}_nbr_probs.npy"`` under ``NBR_DIR``.
    """

    p = NBR_DIR / f"{rows}x{cols}_nbr_probs.npy"
    if p.exists():
        NEIGHBOR_PROBS[(rows, cols)] = np.load(p, mmap_mode="r")


def neighbour_score(
    board_flat: np.ndarray, nbr_probs: np.ndarray, blank_idx: int
) -> np.ndarray:
    """Compute neighbour-based scores for filling a blank cell.

    The function averages conditional probabilities ``P(k | neighbour)``
    for all existing neighbours of the specified blank index.
    """

    rows = int(len(board_flat) ** 0.5)
    cols = len(board_flat) // rows
    r, c = divmod(blank_idx, cols)
    nbr_idx = [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols
    ]
    nbr_vals = [
        board_flat[nr * cols + nc]
        for nr, nc in nbr_idx
        if board_flat[nr * cols + nc] > 0
    ]
    V = nbr_probs.shape[0] - 1
    if not nbr_vals:
        return np.zeros(V + 1, dtype=np.float32)
    probs = nbr_probs[:, nbr_vals]
    return probs.mean(axis=1)
