"""Feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def _board_features(board: np.ndarray) -> np.ndarray:
    """Extract features from a 2D board.

    Parameters
    ----------
    board : np.ndarray
        Board values with ``-1`` representing blanks.

    Returns
    -------
    np.ndarray
        Feature vector with fixed length.
    """
    board = board.astype(float)
    board[board == -1] = np.nan
    global_mean = np.nanmean(board)
    global_std = np.nanstd(board)
    row_means = np.nanmean(board, axis=1)
    row_mean_avg = np.nanmean(row_means)
    row_mean_std = np.nanstd(row_means)
    col_means = np.nanmean(board, axis=0)
    col_mean_avg = np.nanmean(col_means)
    col_mean_std = np.nanstd(col_means)
    filled = np.nan_to_num(board, nan=global_mean)
    _, s, _ = np.linalg.svd(filled, full_matrices=False)
    svd_3 = s[:3]
    local_mean = uniform_filter(filled, size=3)
    local_mean_avg = local_mean.mean()
    local_mean_std = local_mean.std()
    non_duplicate_count = len(np.unique(board[~np.isnan(board)]))
    value_range = np.nanmax(board) - np.nanmin(board)
    features = np.array(
        [
            global_mean,
            global_std,
            row_mean_avg,
            row_mean_std,
            col_mean_avg,
            col_mean_std,
            local_mean_avg,
            local_mean_std,
            non_duplicate_count,
            value_range,
        ]
    )
    features = np.concatenate([features, svd_3])
    return features


def build_features(df: pd.DataFrame, board_shape: tuple[int, int]) -> pd.DataFrame:
    """Build features for the entire dataframe."""
    board_cols = [f"cell_{i}" for i in range(board_shape[0] * board_shape[1])]
    boards = (
        df[board_cols].to_numpy(dtype=float).reshape(-1, board_shape[0], board_shape[1])
    )
    feats = np.vstack([_board_features(b) for b in boards])
    feat_df = pd.DataFrame(feats, columns=[f"f{i}" for i in range(feats.shape[1])])
    return feat_df
