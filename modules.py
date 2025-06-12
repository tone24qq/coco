# modules.py
import numpy as np
import pandas as pd  # 添加 pandas 支援
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
import logging
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from joblib import Parallel, delayed  # 添加並行計算支援

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScratchSolver:
    """
    Solver for analyzing scratch card boards, predicting hidden numbers, and extracting multi-angle features.
    """
    MODULE_REGISTRY: Dict[str, Any] = {}

    def __init__(self):
        self.tree: Optional[cKDTree] = None
        self.known_yx: Optional[np.ndarray] = None
        self.known_vals: Optional[np.ndarray] = None
        self.MODULE_REGISTRY = {
            'compute_dynamic_hot_cold_vectorized': self.compute_dynamic_hot_cold_vectorized,
            'compute_dynamic_hot_cold_advanced': self.compute_dynamic_hot_cold_advanced,
            'idw_vectorized': self.idw_vectorized,
            'compute_block_heatmap_vectorized': self.compute_block_heatmap_vectorized,
            'compute_global_diff_heatmap': self.compute_global_diff_heatmap,
            'compute_focus_score': self.compute_focus_score,
            'detect_skip_patterns': self.detect_skip_patterns,
            'compute_difference_trend': self.compute_difference_trend,
            'detect_mirror_sequences': self.detect_mirror_sequences,
            'connectivity_heatmap': self.connectivity_heatmap,
            'sequence_tail_analyzer': self.sequence_tail_analyzer,
            'analyze_number_patterns': self.analyze_number_patterns
        }
        self.adaptive_weights = AdaptiveWeights({
            "compute_dynamic_hot_cold_vectorized": 0.15,
            "compute_dynamic_hot_cold_advanced": 0.2,
            "compute_block_heatmap_vectorized": 0.1,
            "idw_vectorized": 0.1,
            "compute_global_diff_heatmap": 0.05,
            "compute_focus_score": 0.1,
            "detect_skip_patterns": 0.05,
            "compute_difference_trend": 0.05,
            "detect_mirror_sequences": 0.05,
            "connectivity_heatmap": 0.05,
            "sequence_tail_analyzer": 0.05,
            "analyze_number_patterns": 0.05
        })

    def update_tree(self, grid: np.ndarray) -> None:
        """
        Updates the KDTree with known cell coordinates and values.

        Args:
            grid (np.ndarray): 2D board array.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        self.known_yx = np.argwhere(grid != -1)
        self.known_vals = grid[grid != -1]
        if self.known_yx.size > 0:
            self.tree = cKDTree(self.known_yx)
        else:
            self.tree = None

    def extract_multi_angle_features(self, grid: np.ndarray, output_path: str) -> Dict[str, Any]:
        """
        Extracts features from multiple angles for each number in the grid.

        Args:
            grid (np.ndarray): 2D board array (no hidden cells expected).
            output_path (str): Path to save the features JSON.

        Returns:
            Dict[str, Any]: Dictionary of features (row, column, diagonal, neighborhood).
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        grid_df = pd.DataFrame(grid)  # 使用 pandas 優化數據處理
        features_dict: Dict[str, Any] = {
            "row_features": {},
            "col_features": {},
            "diagonal_features": {},
            "neighborhood_features": {},
            "difference_features": {}
        }

        def process_cell(i, j):
            num = grid[i, j]
            if num == -1:
                return None
            result = {}
            result["row"] = (i, [num])
            result["col"] = (j, [num])
            if i == j:
                result["main_diag"] = ("main", [num])
            if i + j == M - 1:
                result["anti_diag"] = ("anti", [num])
            window = sliding_window_view(
                np.pad(grid, ((1, 1), (1, 1)), mode='edge'), (3, 3)
            )[i, j]
            neighbors = window[window != -1].flatten()
            result["neighbors"] = (f"{i},{j}", neighbors.tolist())
            diffs = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N and grid[ni, nj] != -1:
                    diffs.append(abs(num - grid[ni, nj]))
            result["diffs"] = (f"{i},{j}", diffs)
            return result

        # 使用並行計算提取特徵
        results = Parallel(n_jobs=-1)(
            delayed(process_cell)(i, j) for i in range(M) for j in range(N)
        )
        
        for result in results:
            if result is None:
                continue
            if "row" in result:
                features_dict["row_features"].setdefault(result["row"][0], []).extend(result["row"][1])
            if "col" in result:
                features_dict["col_features"].setdefault(result["col"][0], []).extend(result["col"][1])
            if "main_diag" in result:
                features_dict["diagonal_features"].setdefault(result["main_diag"][0], []).extend(result["main_diag"][1])
            if "anti_diag" in result:
                features_dict["diagonal_features"].setdefault(result["anti_diag"][0], []).extend(result["anti_diag"][1])
            if "neighbors" in result:
                features_dict["neighborhood_features"].setdefault(result["neighbors"][0], []).extend(result["neighbors"][1])
            if "diffs" in result:
                features_dict["difference_features"].setdefault(result["diffs"][0], result["diffs"][1])

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(features_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Features saved to {output_path}")
        except OSError as e:
            logger.error(f"Failed to save features to {output_path}: {e}")

        return features_dict

    def idw_vectorized(self, grid: np.ndarray) -> np.ndarray:
        """
        Computes inverse distance weighting scores for hidden cells.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            np.ndarray: Scores for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        empty_yx = np.argwhere(grid == -1)
        if empty_yx.size == 0:
            return np.array([])
        if self.tree is None or self.known_yx is None or self.known_vals is None:
            return np.full(empty_yx.shape[0], 0.1)
        dists, idxs = self.tree.query(empty_yx, k=min(5, self.known_yx.shape[0]))
        weights = 1.0 / (dists ** 2 + 1e-8)
        est = np.sum(weights * self.known_vals[idxs], axis=1) / np.sum(weights, axis=1)
        return np.where(est < 0.1, 0.1, est)

    def compute_dynamic_hot_cold_vectorized(
        self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'quantile'
    ) -> np.ndarray:
        """
        Computes hot/cold scores based on quantile or std thresholds.

        Args:
            grid (np.ndarray): 2D board array.
            hot_q (float): Hot quantile threshold.
            cold_q (float): Cold quantile threshold.
            method (str): Threshold method ('quantile', 'std').

        Returns:
            np.ndarray: Scores for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        known = grid[grid != -1]
        if known.size == 0:
            return np.full(np.count_nonzero(grid == -1), 0.1)
        if method == 'quantile':
            hot_thr = np.quantile(known, hot_q)
            cold_thr = np.quantile(known, cold_q)
        elif method == 'std':
            mean, std = known.mean(), known.std()
            hot_thr = mean + 1.5 * std
            cold_thr = mean - 1.5 * std
        else:
            hot_thr = known.max() * 0.9
            cold_thr = known.min() * 1.1
        est = self.idw_vectorized(grid)
        est_full = np.zeros_like(grid, dtype=float)
        est_full[grid == -1] = est
        diff_hot = est_full - hot_thr
        diff_cold = cold_thr - est_full
        scores = np.where(
            est_full >= hot_thr,
            np.clip(diff_hot / (hot_thr - cold_thr), 0, 2),
            np.where(
                est_full <= cold_thr,
                -np.clip(diff_cold / (hot_thr - cold_thr), 0, 2),
                0
            )
        )
        scores[grid == -1] = np.where(scores[grid == -1] < 0.1, 0.1, scores[grid == -1])
        return scores[grid == -1]

    def compute_dynamic_hot_cold_advanced(
        self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'quantile'
    ) -> np.ndarray:
        """
        Advanced hot/cold scoring with position and difference weights.

        Args:
            grid (np.ndarray): 2D board array.
            hot_q (float): Hot quantile threshold.
            cold_q (float): Cold quantile threshold.
            method (str): Threshold method ('quantile', 'adaptive').

        Returns:
            np.ndarray: Scores for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        known = grid[grid != -1]
        if known.size == 0:
            return np.full(np.count_nonzero(grid == -1), 0.1)
        
        position_weights = np.exp(-np.sum(np.indices(grid.shape), axis=0) / max(grid.shape))
        diffs = np.abs(np.diff(known))
        diff_weight = np.mean(diffs) if diffs.size > 0 else 1.0
        
        if method == 'adaptive':
            hot_thr = np.percentile(known, 75) + diff_weight
            cold_thr = np.percentile(known, 25) - diff_weight
        else:
            hot_thr = np.quantile(known, hot_q)
            cold_thr = np.quantile(known, cold_q)
        
        est = self.idw_vectorized(grid)
        est_full = np.zeros_like(grid, dtype=float)
        est_full[grid == -1] = est
        diff_hot = est_full - hot_thr
        diff_cold = cold_thr - est_full
        scores = np.where(
            est_full >= hot_thr,
            np.clip(diff_hot / (hot_thr - cold_thr), 0, 2),
            np.where(
                est_full <= cold_thr,
                -np.clip(diff_cold / (hot_thr - cold_thr), 0, 2),
                0
            )
        )
        
        scores[grid == -1] *= position_weights[grid == -1]
        scores[grid == -1] = np.where(scores[grid == -1] < 0.1, 0.1, scores[grid == -1])
        return scores[grid == -1]

    def compute_block_heatmap_vectorized(self, grid: np.ndarray, block_size: int = 2) -> np.ndarray:
        """
        Computes block-based heatmap scores.

        Args:
            grid (np.ndarray): 2D board array.
            block_size (int): Size of sliding block.

        Returns:
            np.ndarray: Scores for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        h, w = grid.shape
        bs = min(block_size, h, w)
        padded = np.pad(grid, ((0, max(0, bs - h)), (0, max(0, bs - w))), mode='edge')
        blocks = sliding_window_view(padded, (bs, bs))
        block_means = np.nanmean(np.where(blocks == -1, np.nan, blocks), axis=(2, 3))
        global_mean = np.nanmean(grid[grid != -1])
        empty = np.argwhere(grid == -1)
        by = empty[:, 0].clip(0, h - bs)
        bx = empty[:, 1].clip(0, w - bs)
        scores = block_means[by, bx] - global_mean
        scores = np.nan_to_num(scores, nan=0.1)
        return np.where(scores < 0.1, 0.1, scores)

    def compute_global_diff_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes global difference heatmap using Laplacian kernel.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        arr = np.where(grid == -1, 0, grid).astype(float)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
        lap = convolve2d(arr, kernel, mode='same', boundary='symm')
        mn, mx = lap.min(), lap.max()
        norm = (lap - mn) / (mx - mn + 1e-8) if mx > mn else lap
        scores = norm[grid == -1]
        scores = np.where(scores < 0.1, 0.1, scores)
        pred = np.full(scores.shape, -1, dtype=int)
        return scores, pred

    def compute_focus_score(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes focus scores based on neighboring cell values.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        mask = (grid != -1).astype(int)
        kernel = np.ones((3, 3)) / 9
        summed = convolve2d(np.where(grid != -1, grid, 0), kernel, mode='same', boundary='symm')
        count = convolve2d(mask, kernel, mode='same', boundary='symm')
        focus_map = summed / (count + 1e-8)
        scores = focus_map[grid == -1]
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=int)

    def detect_skip_patterns(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects arithmetic skip patterns in rows and columns.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for k in range(1, min(4, M, N)):
            for i in range(M):
                windows = sliding_window_view(grid[i], window_shape=k+1)
                for j in range(N - k):
                    if np.all(windows[j] != -1):
                        diff = np.diff(windows[j])
                        if np.all(np.abs(np.diff(diff)) < 1e-10):
                            step = diff[0]
                            for c in range(j+k, N):
                                if grid[i, c] == -1 and 1 <= grid[i, j] + step * (c - j) <= grid.size:
                                    scores[i, c] += 1.0 / k
                                    pred[i, c] = int(grid[i, j] + step * (c - j))
            for j in range(N):
                windows = sliding_window_view(grid[:, j], window_shape=k+1)
                for i in range(M - k):
                    if np.all(windows[i] != -1):
                        diff = np.diff(windows[i])
                        if np.all(np.abs(np.diff(diff)) < 1e-10):
                            step = diff[0]
                            for r in range(i+k, M):
                                if grid[r, j] == -1 and 1 <= grid[i, j] + step * (r - i) <= grid.size:
                                    scores[r, j] += 1.0 / k
                                    pred[r, j] = int(grid[i, j] + step * (r - i))
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores[grid == -1], pred[grid == -1]

    def compute_difference_trend(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes difference trends based on grid gradients.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        d1 = np.diff(grid, axis=1)
        d2 = np.diff(grid, axis=0)
        # 過濾正差值，確保 np.bincount 穩定
        d1_pos = d1[d1 > 0]
        d2_pos = d2[d2 > 0]
        d1_pos_int = np.round(d1_pos).astype(int)
        d2_pos_int = np.round(d2_pos).astype(int)
        max_val = int(np.max(grid[grid != -1])) if np.any(grid != -1) else grid.size
        minlength = max_val + 1
        diff_freq = np.bincount(d1_pos_int, minlength=minlength) + np.bincount(d2_pos_int, minlength=minlength)
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    if j >= 1 and grid[i, j-1] != -1:
                        expected = grid[i, j-1] + 1
                        if 1 <= expected <= grid.size and diff_freq[1] > 0:
                            scores[i, j] = diff_freq[1] / (diff_freq.sum() + 1e-8)
                            pred[i, j] = int(expected)
                    if i >= 1 and grid[i-1, j] != -1:
                        expected = grid[i-1, j] + 1
                        if 1 <= expected <= grid.size and diff_freq[1] > 0:
                            scores[i, j] = max(scores[i, j], diff_freq[1] / (diff_freq.sum() + 1e-8))
                            pred[i, j] = int(expected)
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores[grid == -1], pred[grid == -1]

    def detect_mirror_sequences(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects mirror symmetry patterns in the grid.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.shape}"
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        mid_x = N // 2
        mid_y = M // 2
        left = grid[:, :mid_x]
        right = np.fliplr(grid[:, -mid_x:])
        mirror_lr = np.all((left == right) | (left == -1) | (right == -1), axis=1)
        top = grid[:mid_y, :]
        bottom = np.flipud(grid[-mid_y:, :])
        mirror_ud = np.all((top == bottom) | (top == -1) | (bottom == -1), axis=1)
        diag1 = np.diagonal(grid)
        diag2 = np.diagonal(np.fliplr(grid))
        mirror_diag = np.all((diag1 == diag2) | (diag1 == -1) | (diag2 == -1))
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    # Left-right symmetry
                    if mirror_lr[i] and grid[i, N-1-j] != -1:
                        scores[i, j] = 1.0
                        pred[i, j] = int(grid[i, N-1-j])
                    # Up-down symmetry: 修正索引誤用
                    k = min(i, M-1-i)
                    if k < mid_y and mirror_ud[k] and grid[M-1-i, j] != -1:
                        scores[i, j] = 1.0
                        pred[i, j] = int(grid[M-1-i, j])
                    # Diagonal symmetry
                    if mirror_diag and i == j and grid[M-1-i, M-1-i] != -1:
                        scores[i, j] = 1.0
                        pred[i, j] = int(grid[M-1-i, M-1-i])
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores[grid == -1], pred[grid == -1]

    def connectivity_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes connectivity heatmap based on neighboring cells.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        mask = (grid != -1).astype(np.uint8)
        kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        kernel_8 = np.ones((3, 3)) - np.eye(3)
        conn_4 = convolve2d(mask, kernel_4, mode='same', boundary='symm')
        conn_8-Romance = convolve2d(mask, kernel_8, mode='same', boundary='symm')
        conn_map = (conn_4 + conn_8) / 2
        scores = conn_map[grid == -1]
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=int)

    def sequence_tail_analyzer(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyzes number tails for pattern prediction.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scores and predictions for hidden cells.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        grid_int = np.round(grid).astype(int)  # 確保整數型態
        tails = grid_int[grid_int != -1] % 10
        freq = np.bincount(tails.flatten(), minlength=10) / (np.count_nonzero(grid_int != -1) + 1e-8)
        windows = sliding_window_view(np.pad(grid_int, ((1, 1), (1, 1)), mode='edge'), (3, 3))
        for i in range(M):
            for j in range(N):
                block = windows[i, j]
                block_int = np.round(block).astype(int)
                block_tails = block_int[block_int != -1] % 10
                if block_tails.size > 0:
                    local_freq = np.bincount(block_tails, minlength=10) / (block_tails.size + 1e-8)
                    if grid[i, j] == -1:
                        best_tail = np.argmax(local_freq)
                        scores[i, j] = local_freq[best_tail]
                        candidates = grid_int[grid_int != -1][(grid_int % 10) == best_tail]
                        if candidates.size > 0:
                            pred[i, j] = int(min(candidates) + (best_tail * 10) if min(candidates) < 50 else -1)
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores[grid == -1], pred[grid == -1]

    def analyze_number_patterns(self, grid: np.ndarray) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Analyzes arithmetic patterns in rows and columns.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Dict[Tuple[int, str], Dict[str, Any]]: Detected patterns.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        patterns: Dict[Tuple[int, str], Dict[str, Any]] = {}
        
        def find_arithmetic(arr: np.ndarray, min_len: int = 3) -> Optional[Dict[str, Any]]:
            if len(arr) < min_len:
                return None
            diffs = np.diff(arr)
            if np.all(np.abs(diffs - diffs[0]) < 1e-10):
                return {'type': 'arithmetic', 'diff': diffs[0]}
            return None
        
        grid_df = pd.DataFrame(grid)  # 使用 pandas 優化
        for i in range(M):
            nums = grid_df.iloc[i][grid_df.iloc[i] != -1].values
            if len(nums) >= 3:
                pattern = find_arithmetic(nums)
                if pattern:
                    patterns[(i, 'h')] = pattern
        
        for j in range(N):
            nums = grid_df[j][grid_df[j] != -1].values
            if len(nums) >= 3:
                pattern = find_arithmetic(nums)
                if pattern:
                    patterns[(j, 'v')] = pattern
        
        return patterns

    def pattern_based_prediction(
        self, grid: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts values based on detected patterns.

        Args:
            grid (np.ndarray): 2D board array.
            patterns (Dict): Detected number patterns.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and scores.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        pred = np.full_like(grid, -1, dtype=float)
        scores = np.zeros_like(grid, dtype=float)
        for (idx, direction), pattern in patterns.items():
            if direction == 'h':
                nums = grid[idx][grid[idx] != -1]
                if len(nums) > 0:
                    last_num = nums[-1]
                    diff = pattern['diff']
                    last_idx = np.where(grid[idx] != -1)[0][-1]
                    for j in range(last_idx + 1, N):
                        if grid[idx, j] == -1:
                            predicted = last_num + diff * (j - last_idx)
                            if 1 <= predicted <= grid.size:
                                pred[idx, j] = predicted
                                scores[idx, j] = 1.0
            else:
                nums = grid[:, idx][grid[:, idx] != -1]
                if len(nums) > 0:
                    last_num = nums[-1]
                    diff = pattern['diff']
                    last_idx = np.where(grid[:, idx] != -1)[0][-1]
                    for i in range(last_idx + 1, M):
                        if grid[i, idx] == -1:
                            predicted = last_num + diff * (i - last_idx)
                            if 1 <= predicted <= grid.size:
                                pred[i, idx] = predicted
                                scores[i, idx] = 1.0
        scores = np.where(scores < 0.1, 0.1, scores)
        return pred, scores

    def local_relationship_prediction(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts values based on local neighbor relationships.

        Args:
            grid (np.ndarray): 2D board array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and scores.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        M, N = grid.shape
        pred = np.full_like(grid, -1, dtype=float)
        scores = np.zeros_like(grid, dtype=float)
        kernel = np.ones((3, 3)) / 8
        neighbor_sum = convolve2d(np.where(grid != -1, grid, 0), kernel, mode='same', boundary='symm')
        neighbor_count = convolve2d(grid != -1, kernel, mode='same', boundary='symm')
        pred[grid == -1] = neighbor_sum[grid == -1] / (neighbor_count[grid == -1] + 1e-8)
        scores[grid == -1] = neighbor_count[grid == -1] / 8
        pred[grid == -1] = np.clip(pred[grid == -1], 1, grid.size)
        scores = np.where(scores < 0.1, 0.1, scores)
        return pred, scores

    def heatmap_based_prediction(self, grid: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates predictions based on heatmap scores.

        Args:
            grid (np.ndarray): 2D board array.
            scores (np.ndarray): Scores for hidden cells.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and confidence scores.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        pred = np.zeros_like(grid, dtype=float)
        confidence = np.zeros_like(grid, dtype=float)
        empty_yx = np.argwhere(grid == -1)
        if len(scores) == len(empty_yx):
            pred[empty_yx[:, 0], empty_yx[:, 1]] = scores
            confidence[empty_yx[:, 0], empty_yx[:, 1]] = scores
        pred = np.clip(pred, 1, grid.size)
        confidence = np.where(confidence < 0.1, 0.1, confidence)
        return pred, confidence

    def integrate_predictions(
        self, grid: np.ndarray, scores: np.ndarray, patterns: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates multiple prediction methods.

        Args:
            grid (np.ndarray): 2D board array.
            scores (np.ndarray): Scores for hidden cells.
            patterns (Dict): Detected number patterns.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Final predictions and confidence scores.
        """
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        predictions = np.full_like(grid, -1, dtype=float)
        confidence = np.zeros_like(grid, dtype=float)
        
        pattern_pred, pattern_scores = self.pattern_based_prediction(grid, patterns)
        heatmap_pred, heatmap_scores = self.heatmap_based_prediction(grid, scores)
        local_pred, local_scores = self.local_relationship_prediction(grid)
        
        w_pattern = 0.4
        w_heatmap = 0.4
        w_local = 0.2
        
        empty_mask = (grid == -1)
        empty_yx = np.where(empty_mask)
        for i, j in zip(empty_yx[0], empty_yx[1]):
            predictions