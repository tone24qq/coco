"""Feature engineering utilities."""

from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.utils.extmath import randomized_svd


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
    _, s, _ = randomized_svd(filled, n_components=4, random_state=0)
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


def build_features(
    df: pd.DataFrame, board_shape: tuple[int, int], n_jobs: int = 1
) -> pd.DataFrame:
    """Build features for the entire dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing board cells.
    board_shape : tuple[int, int]
        Shape of the board (rows, cols).
    n_jobs : int, optional
        Number of worker processes for feature extraction, by default ``1``.
    """
    board_cols = [f"cell_{i}" for i in range(board_shape[0] * board_shape[1])]
    boards = (
        df[board_cols].to_numpy(dtype=float).reshape(-1, board_shape[0], board_shape[1])
    )
    if n_jobs == 1:
        feats = np.vstack([_board_features(b) for b in boards])
    else:
        with Pool(n_jobs) as pool:
            feats = np.vstack(
                list(pool.imap_unordered(_board_features, boards, chunksize=500))
            )
    feat_df = pd.DataFrame(feats, columns=[f"f{i}" for i in range(feats.shape[1])])
    return feat_df
