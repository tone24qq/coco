# modules.py (continued in script.py)
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)

class ScratchSolver:
    """
    Scratch card analysis solver with multiple prediction modules.

    This class implements various analysis techniques to predict hidden numbers
    in a scratch card grid, supporting a maximum grid size of 20x20.
    """
    MODULE_REGISTRY = {}

    def __init__(self):
        """
        Initialize the ScratchSolver with empty data structures.
        """
        self.tree = None
        self.known_yx = None
        self.known_vals = None
        self.MODULE_REGISTRY = {
            'compute_dynamic_hot_cold_vectorized': self.compute_dynamic_hot_cold_vectorized,
            'idw_vectorized': self.idw_vectorized
        }

    def update_tree(self, grid):
        """
        Update the KDTree with known positions and values.

        Args:
            grid (np.ndarray): 2D array of the scratch card grid.
        """
        self.known_yx = np.argwhere(grid != -1)
        self.known_vals = grid[grid != -1]
        if self.known_yx.size > 0:
            self.tree = cKDTree(self.known_yx)
        else:
            self.tree = None

    def idw_vectorized(self, grid):
        """
        Inverse Distance Weighting (IDW) prediction for empty cells.

        Args:
            grid (np.ndarray): 2D array of the scratch card grid.

        Returns:
            np.ndarray: Predicted values for empty cells.
        """
        empty_yx = np.argwhere(grid == -1)
        if empty_yx.size == 0:
            return np.array([])
        if self.tree is None or self.known_yx is None:
            return np.zeros(empty_yx.shape[0]) / empty_yx.shape[0]
        dists, idxs = self.tree.query(empty_yx, k=min(5, self.known_yx.shape[0]))
        weights = 1.0 / (dists ** 2 + 1e-8)
        est = np.sum(weights * self.known_vals[idxs], axis=1) / np.sum(weights, axis=1)
        return est

    def compute_dynamic_hot_cold_vectorized(self, grid, hot_q=0.9, cold_q=0.1, method='quantile'):
        """
        Compute dynamic hot/cold scores for empty cells.

        Args:
            grid (np.ndarray): 2D array of the scratch card grid.
            hot_q (float): Quantile for hot threshold.
            cold_q (float): Quantile for cold threshold.
            method (str): Method for threshold calculation ('quantile', 'std', 'minmax').

        Returns:
            np.ndarray: Scores for empty cells.
        """
        known = grid[grid != -1]
        if known.size == 0:
            return np.zeros(np.count_nonzero(grid == -1))
        if method == 'quantile':
            hot_thr = np.quantile(known, hot_q)
            cold_thr = np.quantile(known, cold_q)
        elif method == 'std':
            mean, std = known.mean(), known.std()
            hot_thr = mean + 1.5 * std
            cold_thr = mean - 1.5 * std
        else:  # minmax
            hot_thr = known.max() * 0.9
            cold_thr = known.min() * 1.1
        est = self.idw_vectorized(grid)
        est_full = np.zeros_like(grid, dtype=float)
        est_full[grid == -1] = est
        diff_hot = est_full - hot_thr
        diff_cold = cold_thr - est_full
        scores = np.where(est_full >= hot_thr, np.clip(diff_hot / (hot_thr - cold_thr), 0, 2),
                         np.where(est_full <= cold_thr, -np.clip(diff_cold / (hot_thr - cold_thr), 0, 2), 0))
        return scores[grid == -1]

    def classify_board_type(self, dynamic_scores, hot_thresh=0.5, cold_thresh=-0.5):
        """
        Classify the board type based on dynamic scores.

        Args:
            dynamic_scores (np.ndarray): Scores from dynamic hot/cold analysis.
            hot_thresh (float): Threshold for hot classification.
            cold_thresh (float): Threshold for cold classification.

        Returns:
            str: Board type ('HOT', 'COLD', or 'UNIFORM').
        """
        total = dynamic_scores.sum() / (np.count_nonzero(dynamic_scores != 0) + 1e-8)
        if total >= hot_thresh:
            return 'HOT'
        elif total <= cold_thresh:
            return 'COLD'
        else:
            return 'UNIFORM'

    def fuse_scores_vectorized(self, mod_scores, board_type, default_weights):
        """
        Fuse scores from multiple modules with type-specific weighting.

        Args:
            mod_scores (dict): Dictionary of module scores.
            board_type (str): Classified board type.
            default_weights (dict): Default weights for modules.

        Returns:
            np.ndarray: Fused scores.
        """
        w = default_weights.copy()
        if board_type == 'HOT':
            w['compute_dynamic_hot_cold_vectorized'] *= 1.5
            w['idw_vectorized'] *= 1.2
        elif board_type == 'COLD':
            w['idw_vectorized'] *= 1.3
        names = list(mod_scores.keys())
        score_mat = np.stack([mod_scores[n] for n in names], axis=1)
        weight_arr = np.array([w.get(n, 0.1) for n in names])
        heat_factor = np.abs(mod_scores.get('compute_dynamic_hot_cold_vectorized', np.zeros(score_mat.shape[0])).sum()) / (score_mat.shape[0] + 1e-8)
        final = (score_mat.dot(weight_arr) / (weight_arr.sum() + 1e-8)) * (1 + heat_factor * 0.5)
        return final

    def predict_top3_vectorized(self, final_scores, empty_positions):
        """
        Predict the top 3 positions based on final scores.

        Args:
            final_scores (np.ndarray): Fused scores for empty positions.
            empty_positions (np.ndarray): Coordinates of empty cells.

        Returns:
            list: Top 3 positions with scores and contributions.
        """
        idxs = np.argsort(-final_scores)[:3]
        unique_idx = np.unique(idxs, return_index=True)[1]
        top3_idx = idxs[np.sort(unique_idx)[:3]]
        contributions = {name: float(final_scores[i]) for i, name in enumerate(self.MODULE_REGISTRY.keys()) if i in top3_idx}
        return [(int(empty_positions[i][0]), int(empty_positions[i][1]), float(final_scores[i]), contributions) for i in top3_idx]