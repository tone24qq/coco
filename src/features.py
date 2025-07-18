"""Feature extraction for scratch-card boards."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import svd
from scipy.ndimage import uniform_filter


def _board_features(board: ArrayLike) -> np.ndarray:
    """Compute board features with fixed dimension.

    Parameters
    ----------
    board:
        2D array representing the board with ``-1`` for blanks.

    Returns
    -------
    numpy.ndarray
        Feature vector of consistent length.
    """
    arr = np.asarray(board, dtype=float)
    mask = arr == -1
    arr[mask] = 0

    # Row/column statistics
    row_mean = arr.mean(axis=1)
    row_std = arr.std(axis=1)
    col_mean = arr.mean(axis=0)
    col_std = arr.std(axis=0)

    # Singular values
    u, s, vt = svd(arr, full_matrices=False)
    svd_features = s[:3]

    # Local statistics using 3x3 uniform filter
    local_mean = uniform_filter(arr, size=3)
    local_std = np.sqrt(uniform_filter(arr**2, size=3) - local_mean**2)
    loc_features = np.array([local_mean.mean(), local_std.mean()])

    # Unique count and range check
    unique_count = np.unique(arr[~mask]).size
    value_range = arr.max() - arr.min()

    features = np.concatenate(
        [
            row_mean,
            row_std,
            col_mean,
            col_std,
            svd_features,
            loc_features,
            [unique_count, value_range],
        ]
    )
    return features
