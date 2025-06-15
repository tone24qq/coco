# modules.py

import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
import pandas as pd
import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from joblib import Parallel, delayed

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/modules.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveWeights:
    """
    Manages adaptive weights for module scores.

    Attributes:
        weights (Dict[str, float]): Current module weights.
        history (List[Dict[str, Any]]): History of weights and success rates.
    """
    def __init__(self, initial_weights: Dict[str, float]):
        """
        Initialize with given weights.

        Args:
            initial_weights (Dict[str, float]): Initial module weights.
        """
        self.weights = initial_weights.copy()
        self.history: List[Dict[str, Any]] = []

    def update(self, success_rate: float, module_scores: Dict[str, np.ndarray]) -> None:
        """
        Update weights based on prediction success rate and module scores.

        Args:
            success_rate (float): Prediction success rate.
            module_scores (Dict[str, np.ndarray]): Module scores.

        Raises:
            ValueError: If success_rate is invalid.
        """
        try:
            if not 0 <= success_rate <= 1:
                raise ValueError(f"Invalid success_rate {success_rate}, must be between 0 and 1")

            alpha = 0.1
            self.history.append({
                'success_rate': success_rate,
                'weights': self.weights.copy(),
                'scores': {k: v.tolist() for k, v in module_scores.items()}
            })

            if len(self.history) >= 5:
                best_config = max(self.history[-5:], key=lambda x: x['success_rate'])
                for key in self.weights:
                    self.weights[key] += alpha * (best_config['weights'][key] - self.weights[key])
                total = sum(self.weights.values())
                self.weights = {k: v/total for k, v in self.weights.items()}
            logger.debug("Updated weights: %s", self.weights)
        except ValueError as e:
            logger.error("Failed to update weights: %s", e)
            raise

    def save_history(self, filepath: str) -> None:
        """
        Save weight history to a JSON file.

        Args:
            filepath (str): Path to save history.

        Raises:
            OSError: If file saving fails.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.info("Weight history saved to %s", filepath)
        except OSError as e:
            logger.error("Failed to save weight history to %s: %s", filepath, e)
            raise

    def load_history(self, filepath: str) -> None:
        """
        Load weight history from a JSON file.

        Args:
            filepath (str): Path to load history from.

        Raises:
            OSError: If file access fails.
            json.JSONDecodeError: If JSON parsing fails.
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info("Loaded weight history from %s", filepath)
            else:
                logger.warning("History file %s does not exist", filepath)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load weight history from %s: %s", filepath, e)
            raise

class ScratchSolver:
    """
    Scratch card analysis module for feature extraction and hidden number prediction.

    Supports precise detection for grids of any size.

    Attributes:
        tree (Optional[cKDTree]): KDTree storing known cell coordinates.
        known_yx (Optional[np.ndarray]): Known cell coordinates.
        known_vals (Optional[np.ndarray]): Known cell values.
        MODULE_REGISTRY (Dict[str, Any]): Registered analysis modules.
        adaptive_weights (AdaptiveWeights): Adaptive weight manager.
    """
    MODULE_REGISTRY: Dict[str, Any] = {}

    def __init__(self):
        """
        Initialize the solver with module registry and adaptive weights.
        """
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
            "compute_dynamic_hot_cold_advanced": 0.5,
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
        Update KDTree to store coordinates and values of known cells.

        Args:
            grid (np.ndarray): 2D grid array.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            self.known_yx = np.argwhere(grid != -1)
            self.known_vals = grid[grid != -1]
            if self.known_yx.size > 0:
                self.tree = cKDTree(self.known_yx)
                logger.debug("Initialized KDTree with %d known points", self.known_yx.size)
            else:
                self.tree = None
                self.known_yx = None
                self.known_vals = None
                logger.info("No KDTree initialized: no known cells (all -1)")
        except AssertionError as e:
            logger.error("Failed to update KDTree: %s", e)
            raise

    def extract_multi_angle_features(self, grid: np.ndarray, output_path: str) -> Dict[str, Any]:
        """
        Extract grid features from multiple angles and save to a file.

        Args:
            grid (np.ndarray): 2D grid array.
            output_path (str): Path to save features.

        Returns:
            Dict[str, Any]: Extracted feature dictionary.

        Raises:
            AssertionError: If grid is not 2D.
            OSError: If feature saving fails.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            features_dict: Dict[str, Any] = {
                "row_features": {},
                "col_features": {},
                "diagonal_features": {},
                "neighborhood_features": {},
                "difference_features": {}
            }

            grid_df = pd.DataFrame(grid)
            for i in range(M):
                row = grid_df.iloc[i][grid_df.iloc[i] != -1]
                features_dict["row_features"][i] = row.tolist()

            for j in range(N):
                col = grid_df[j][grid_df[j] != -1]
                features_dict["col_features"][j] = col.tolist()

            diag = np.diagonal(grid)
            anti_diag = np.diagonal(np.fliplr(grid))
            features_dict["diagonal_features"]["main"] = diag[diag != -1].tolist()
            features_dict["diagonal_features"]["anti"] = anti_diag[anti_diag != -1].tolist()

            def process_cell(i: int, j: int) -> Optional[Dict[str, Any]]:
                if grid[i, j] == -1:
                    return None
                try:
                    window = sliding_window_view(
                        np.pad(grid, ((1, 1), (1, 1)), mode='edge'), (3, 3)
                    )[i, j]
                    neighbors = window[window != -1].flatten().tolist()
                    diffs = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < M and 0 <= nj < N and grid[ni, nj] != -1:
                            diffs.append(abs(grid[i, j] - grid[ni, nj]))
                    return {f"{i},{j}": {"neighbors": neighbors, "diffs": diffs}}
                except ValueError as e:
                    logger.warning("Failed to process cell (%d, %d): %s", i, j, e)
                    return None

            results = Parallel(n_jobs=1)(
                delayed(process_cell)(i, j) for i in range(M) for j in range(N)
            )
            for result in results:
                if result:
                    for key, value in result.items():
                        features_dict["neighborhood_features"][key] = value["neighbors"]
                        features_dict["difference_features"][key] = value["diffs"]

            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(features_dict, f, ensure_ascii=False, indent=2)
                logger.info("Features saved to %s", output_path)
            except OSError as e:
                logger.error("Failed to save features to %s: %s", output_path, e)
                raise

            logger.info("Extracted features for 1 grid, shape %s", grid.shape)
            return features_dict
        except AssertionError as e:
            logger.error("Failed to extract features: %s", e)
            raise

    def idw_vectorized(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute inverse distance weighted scores for hidden cells.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            np.ndarray: Scores for hidden cells.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            empty_yx = np.argwhere(grid == -1)
            if empty_yx.size == 0:
                return np.array([])
            if self.tree is None or self.known_yx is None or self.known_vals is None:
                logger.debug("No KDTree initialized, returning default scores")
                return np.full(empty_yx.shape[0], 0.1)
            dists, idxs = self.tree.query(empty_yx, k=min(5, self.known_yx.shape[0]))
            weights = 1.0 / (dists ** 2 + 1e-8)
            est = np.sum(weights * self.known_vals[idxs], axis=1) / np.sum(weights, axis=1)
            return np.where(est < 0.1, 0.1, est)
        except AssertionError as e:
            logger.error("Failed to compute IDW scores: %s", e)
            raise

    def compute_dynamic_hot_cold_vectorized(
        self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'quantile'
    ) -> np.ndarray:
        """
        Compute hot-cold scores based on quantiles or standard deviation.

        Args:
            grid (np.ndarray): 2D grid array.
            hot_q (float): Hot quantile threshold.
            cold_q (float): Cold quantile threshold.
            method (str): Threshold method ('quantile', 'std').

        Returns:
            np.ndarray: Hot-cold scores for hidden cells.

        Raises:
            AssertionError: If grid is not 2D.
            ValueError: If quantiles or method are invalid.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            if not 0 <= hot_q <= 1 or not 0 <= cold_q <= 1:
                raise ValueError(f"Invalid quantiles: hot_q={hot_q}, cold_q={cold_q}")
            if method not in ['quantile', 'std']:
                raise ValueError(f"Invalid method: {method}")

            grid = grid.astype(np.int64)
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
        except (AssertionError, ValueError) as e:
            logger.error("Failed to compute hot-cold scores: %s", e)
            raise

    def compute_dynamic_hot_cold_advanced(
        self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'adaptive'
    ) -> np.ndarray:
        """
        Compute advanced hot-cold scores considering position, difference, and connectivity weights.

        Args:
            grid (np.ndarray): 2D grid array.
            hot_q (float): Hot quantile threshold.
            cold_q (float): Cold quantile threshold.
            method (str): Threshold method ('adaptive', 'quantile').

        Returns:
            np.ndarray: Advanced hot-cold scores for hidden cells.

        Raises:
            AssertionError: If grid is not 2D.
            ValueError: If quantiles or method are invalid.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            if not 0 <= hot_q <= 1 or not 0 <= cold_q <= 1:
                raise ValueError(f"Invalid quantiles: hot_q={hot_q}, cold_q={cold_q}")
            if method not in ['adaptive', 'quantile']:
                raise ValueError(f"Invalid method: {method}")

            grid = grid.astype(np.int64)
            known = grid[grid != -1]
            if known.size == 0:
                return np.full(np.count_nonzero(grid == -1), 0.1)

            position_weights = np.exp(-np.sum(np.indices(grid.shape), axis=0) / max(grid.shape))

            diffs = np.abs(np.diff(known))
            diff_weight = np.mean(diffs) if diffs.size > 0 else 1.0

            conn_scores, _ = self.connectivity_heatmap(grid)

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

            empty_yx = np.argwhere(grid == -1)
            for idx, (i, j) in enumerate(empty_yx):
                scores[idx] *= position_weights[i, j] * (1 + conn_scores[idx] * 0.5)

            scores = np.where(scores < 0.1, 0.1, scores)
            return scores
        except (AssertionError, ValueError) as e:
            logger.error("Failed to compute advanced hot-cold scores: %s", e)
            raise

    def compute_block_heatmap_vectorized(self, grid: np.ndarray, block_size: int = 2) -> np.ndarray:
        """
        Compute block-based heatmap scores.

        Args:
            grid (np.ndarray): 2D grid array.
            block_size (int): Size of analysis blocks.

        Returns:
            np.ndarray: Block scores for hidden cells.

        Raises:
            AssertionError: If grid is not 2D.
            ValueError: If block_size is invalid.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            if block_size < 1:
                raise ValueError(f"Invalid block_size {block_size}, must be positive")

            grid = grid.astype(np.int64)
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
        except (AssertionError, ValueError) as e:
            logger.error("Failed to compute block heatmap: %s", e)
            raise

    def compute_global_diff_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute global difference heatmap using Laplacian kernel.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            arr = np.where(grid == -1, 0, grid).astype(float)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            lap = convolve2d(arr, kernel, mode='same', boundary='symm')
            mn, mx = lap.min(), lap.max()
            norm = (lap - mn) / (mx - mn + 1e-8) if mx > mn else lap
            scores = norm[grid == -1]
            scores = np.where(scores < 0.1, 0.1, scores)
            pred = np.full(scores.shape, -1, dtype=np.int64)
            return scores, pred
        except AssertionError as e:
            logger.error("Failed to compute global diff heatmap: %s", e)
            raise

    def compute_focus_score(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute focus scores based on neighbor cell values.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            mask = (grid != -1).astype(np.int64)
            kernel = np.ones((3, 3)) / 9
            summed = convolve2d(np.where(grid != -1, grid, 0), kernel, mode='same', boundary='symm')
            count = convolve2d(mask, kernel, mode='same', boundary='symm')
            focus_map = summed / (count + 1e-8)
            scores = focus_map[grid == -1]
            scores = np.where(scores < 0.1, 0.1, scores)
            return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=np.int64)
        except AssertionError as e:
            logger.error("Failed to compute focus scores: %s", e)
            raise

    def detect_skip_patterns(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect arithmetic skip patterns in rows and columns.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            scores = np.zeros((M, N), dtype=float)
            pred = np.full((M, N), -1, dtype=np.int64)
            for k in range(1, min(4, M, N)):
                for i in range(M):
                    windows = sliding_window_view(grid[i], window_shape=k+1)
                    for j in range(N - k):
                        if np.all(windows[j] != -1):
                            diff = np.diff(windows[j])
                            if np.all(np.abs(diff - diff[0]) < 1e-10):
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
                            if np.all(np.abs(diff - diff[0]) < 1e-10):
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
        except AssertionError as e:
            logger.error("Failed to detect skip patterns: %s", e)
            raise

    def compute_difference_trend(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute difference trend based on grid gradients.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            scores = np.zeros((M, N), dtype=float)
            pred = np.full((M, N), -1, dtype=np.int64)
            d1 = np.diff(grid, axis=1)
            d2 = np.diff(grid, axis=0)
            d1_pos = d1[d1 > 0]
            d2_pos = d2[d2 > 0]
            d1_pos_int = np.round(d1_pos).astype(np.int64)
            d2_pos_int = np.round(d2_pos).astype(np.int64)
            max_val = int(np.max(grid[grid != -1])) if np.any(grid != -1) else grid.size
            minlength = max_val + 1
            diff_freq = np.zeros(minlength, dtype=np.int64)
            if d1_pos_int.size > 0:
                diff_freq += np.bincount(d1_pos_int, minlength=minlength)
            if d2_pos_int.size > 0:
                diff_freq += np.bincount(d2_pos_int, minlength=minlength)
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
        except AssertionError as e:
            logger.error("Failed to compute difference trend: %s", e)
            raise

    def detect_mirror_sequences(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect mirror symmetry patterns in the grid.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            scores = np.zeros((M, N), dtype=float)
            pred = np.full((M, N), -1, dtype=np.int64)
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
                        if mirror_lr[i] and grid[i, N-1-j] != -1:
                            scores[i, j] = 1.0
                            pred[i, j] = int(grid[i, N-1-j])
                        k = min(i, M-1-i)
                        if k < mid_y and mirror_ud[k] and grid[M-1-i, j] != -1:
                            scores[i, j] = 1.0
                            pred[i, j] = int(grid[M-1-i, j])
                        if mirror_diag and i == j and grid[M-1-i, M-1-i] != -1:
                            scores[i, j] = 1.0
                            pred[i, j] = int(grid[M-1-i, M-1-i])
            scores[grid != -1] = 0
            mn, mx = scores.min(), scores.max()
            scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
            scores = np.where(scores < 0.1, 0.1, scores)
            return scores[grid == -1], pred[grid == -1]
        except AssertionError as e:
            logger.error("Failed to detect mirror sequences: %s", e)
            raise

    def connectivity_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute connectivity heatmap based on neighbor cells.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            mask = (grid != -1).astype(np.uint8)
            kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            kernel_8 = np.ones((3, 3)) - np.eye(3)
            conn_4 = convolve2d(mask, kernel_4, mode='same', boundary='symm')
            conn_8 = convolve2d(mask, kernel_8, mode='same', boundary='symm')
            conn_map = (conn_4 + conn_8) / 2
            scores = conn_map[grid == -1]
            scores = np.where(scores < 0.1, 0.1, scores)
            return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=np.int64)
        except AssertionError as e:
            logger.error("Failed to compute connectivity heatmap: %s", e)
            raise

    def sequence_tail_analyzer(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze number tails to predict patterns.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell scores and predictions.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            scores = np.zeros((M, N), dtype=float)
            pred = np.full((M, N), -1, dtype=np.int64)
            tails = grid[grid != -1] % 10
            freq = np.bincount(tails.flatten(), minlength=10) / (np.count_nonzero(grid != -1) + 1e-8)
            windows = sliding_window_view(np.pad(grid, ((1, 1), (1, 1)), mode='edge'), (3, 3))
            for i in range(M):
                for j in range(N):
                    block = windows[i, j]
                    block_tails = block[block != -1] % 10
                    if block_tails.size > 0:
                        local_freq = np.bincount(block_tails, minlength=10) / (block_tails.size + 1e-8)
                        if grid[i, j] == -1:
                            best_tail = np.argmax(local_freq)
                            scores[i, j] = local_freq[best_tail]
                            candidates = grid[grid != -1][(grid[grid != -1] % 10) == best_tail]
                            if candidates.size > 0:
                                pred[i, j] = int(min(candidates) + (best_tail * 10) if min(candidates) < 50 else -1)
            scores[grid != -1] = 0
            mn, mx = scores.min(), scores.max()
            scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
            scores = np.where(scores < 0.1, 0.1, scores)
            return scores[grid == -1], pred[grid == -1]
        except AssertionError as e:
            logger.error("Failed to analyze sequence tails: %s", e)
            raise

    def analyze_number_patterns(self, grid: np.ndarray) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Analyze arithmetic patterns in rows and columns.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Dict[Tuple[int, str], Dict[str, Any]]: Detected patterns.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            M, N = grid.shape
            patterns: Dict[Tuple[int, str], Dict[str, Any]] = {}

            def find_arithmetic(arr: np.ndarray, min_len: int = 3) -> Optional[Dict[str, Any]]:
                if len(arr) < min_len:
                    return None
                diffs = np.diff(arr)
                if np.all(np.abs(diffs - diffs[0]) < 1e-10):
                    return {'type': 'arithmetic', 'diff': diffs[0]}
                for k in range(2, min(5, len(arr))):
                    sub_diffs = np.diff(arr[::k])
                    if len(sub_diffs) > 1 and np.all(np.abs(sub_diffs - sub_diffs[0]) < 1e-10):
                        return {'type': 'skip_arithmetic', 'diff': sub_diffs[0], 'skip': k}
                return None

            def process_row(i: int) -> Optional[Tuple[Tuple[int, str], Dict[str, Any]]]:
                nums = grid[i][grid[i] != -1]
                if len(nums) >= 3:
                    pattern = find_arithmetic(nums)
                    if pattern:
                        return (i, 'h'), pattern
                return None

            def process_col(j: int) -> Optional[Tuple[Tuple[int, str], Dict[str, Any]]]:
                nums = grid[:, j][grid[:, j] != -1]
                if len(nums) >= 3:
                    pattern = find_arithmetic(nums)
                    if pattern:
                        return (j, 'v'), pattern
                return None

            row_results = Parallel(n_jobs=1)(delayed(process_row)(i) for i in range(M))
            col_results = Parallel(n_jobs=1)(delayed(process_col)(j) for j in range(N))

            for result in row_results + col_results:
                if result:
                    patterns[result[0]] = result[1]

            return patterns
        except AssertionError as e:
            logger.error("Failed to analyze number patterns: %s", e)
            raise

    def pattern_based_prediction(
        self, grid: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict based on detected patterns.

        Args:
            grid (np.ndarray): 2D grid array.
            patterns (Dict[Tuple[int, str], Dict[str, Any]]): Detected patterns.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell predictions and scores.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
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
                        skip = pattern.get('skip', 1)
                        for j in range(last_idx + skip, N, skip):
                            if grid[idx, j] == -1:
                                predicted = last_num + diff * ((j - last_idx) // skip)
                                if 1 <= predicted <= grid.size:
                                    pred[idx, j] = predicted
                                    scores[idx, j] = 1.0
                else:
                    nums = grid[:, idx][grid[:, idx] != -1]
                    if len(nums) > 0:
                        last_num = nums[-1]
                        diff = pattern['diff']
                        last_idx = np.where(grid[:, idx] != -1)[0][-1]
                        skip = pattern.get('skip', 1)
                        for i in range(last_idx + skip, M, skip):
                            if grid[i, idx] == -1:
                                predicted = last_num + diff * ((i - last_idx) // skip)
                                if 1 <= predicted <= grid.size:
                                    pred[i, idx] = predicted
                                    scores[i, idx] = 1.0
            scores = np.where(scores < 0.1, 0.1, scores)
            return pred, scores
        except AssertionError as e:
            logger.error("Failed to perform pattern-based prediction: %s", e)
            raise

    def local_relationship_prediction(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict based on neighbor relationships.

        Args:
            grid (np.ndarray): 2D grid array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell predictions and scores.

        Raises:
            AssertionError: If grid is not 2D.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
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
        except AssertionError as e:
            logger.error("Failed to perform local relationship prediction: %s", e)
            raise

    def heatmap_based_prediction(self, grid: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions based on heatmap scores.

        Args:
            grid (np.ndarray): 2D grid array.
            scores (np.ndarray): Heatmap scores.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Hidden cell predictions and confidence scores.

        Raises:
            AssertionError: If grid is not 2D.
            ValueError: If scores shape is invalid.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            empty_yx = np.argwhere(grid == -1)
            if len(scores) != len(empty_yx):
                raise ValueError(f"Scores length {len(scores)} does not match hidden cells {len(empty_yx)}")

            pred = np.zeros_like(grid, dtype=float)
            confidence = np.zeros_like(grid, dtype=float)
            pred[empty_yx[:, 0], empty_yx[:, 1]] = scores
            confidence[empty_yx[:, 0], empty_yx[:, 1]] = scores
            pred = np.clip(pred, 1, grid.size)
            confidence = np.where(confidence < 0.1, 0.1, confidence)
            return pred, confidence
        except (AssertionError, ValueError) as e:
            logger.error("Failed to perform heatmap-based prediction: %s", e)
            raise

    def integrate_predictions(
        self, grid: np.ndarray, scores: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate multiple prediction methods.

        Args:
            grid (np.ndarray): 2D grid array.
            scores (np.ndarray): Heatmap scores.
            patterns (Dict[Tuple[int, str], Dict[str, Any]]): Detected patterns.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Integrated predictions and confidence scores.

        Raises:
            AssertionError: If grid is not 2D.
            ValueError: If scores shape is invalid.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            grid = grid.astype(np.int64)
            empty_yx = np.argwhere(grid == -1)
            if len(scores) != len(empty_yx):
                raise ValueError(f"Scores length {len(scores)} does not match hidden cells {len(empty_yx)}")

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
                predictions[i, j] = (
                    pattern_pred[i, j] * w_pattern +
                    heatmap_pred[i, j] * w_heatmap +
                    local_pred[i, j] * w_local
                )
                confidence[i, j] = max(pattern_scores[i, j], heatmap_scores[i, j], local_scores[i, j])
                predictions[i, j] = np.clip(predictions[i, j], 1, grid.size)
                confidence[i, j] = max(confidence[i, j], 0.1)

            return predictions, confidence
        except (AssertionError, ValueError) as e:
            logger.error("Failed to integrate predictions: %s", e)
            raise

    def evaluate_prediction(
        self, grid: np.ndarray, prediction: np.ndarray, true_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction accuracy and pattern matching.

        Args:
            grid (np.ndarray): Original grid.
            prediction (np.ndarray): Predicted values.
            true_values (np.ndarray): True values.

        Returns:
            Dict[str, float]: Evaluation metrics.

        Raises:
            AssertionError: If inputs are not 2D or shapes mismatch.
        """
        try:
            assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array, shape {grid.shape}"
            assert prediction.ndim == 2, f"Expected 2D prediction, got {prediction.ndim}D array, shape {prediction.shape}"
            assert true_values.ndim == 2, f"Expected 2D true values, got {true_values.ndim}D array, shape {true_values.shape}"
            assert prediction.shape == grid.shape == true_values.shape, "Shapes mismatch"

            grid = grid.astype(np.int64)
            metrics = {
                'accuracy': 0.0,
                'pattern_match': 0.0,
                'value_diff': 0.0
            }

            mask = (grid == -1)
            if np.any(mask):
                correct = (prediction[mask] == true_values[mask])
                metrics['accuracy'] = correct.mean() if correct.size > 0 else 0.0
                metrics['value_diff'] = np.abs(prediction[mask] - true_values[mask]).mean() if correct.size > 0 else 0.0

            pred_patterns = self.analyze_number_patterns(prediction)
            true_patterns = self.analyze_number_patterns(true_values)
            metrics['pattern_match'] = len(
                set(pred_patterns.keys()) & set(true_patterns.keys())
            ) / max(len(pred_patterns), len(true_patterns), 1)

            return metrics
        except AssertionError as e:
            logger.error("Failed to evaluate prediction: %s", e)
            raise

    def classify_board_type(self, dynamic_scores: np.ndarray, hot_thresh: float = 0.5, cold_thresh: float = -0.5) -> str:
        """
        Classify grid as hot, cold, or uniform based on scores.

        Args:
            dynamic_scores (np.ndarray): Dynamic scores array.
            hot_thresh (float): Hot threshold.
            cold_thresh (float): Cold threshold.

        Returns:
            str: Grid type ('HOT', 'COLD', 'UNIFORM').

        Raises:
            ValueError: If thresholds are invalid.
        """
        try:
            if hot_thresh <= cold_thresh:
                raise ValueError(f"Hot threshold {hot_thresh} must be greater than cold threshold {cold_thresh}")

            total = dynamic_scores.sum() / (np.count_nonzero(dynamic_scores != 0) + 1e-8)
            if total >= hot_thresh:
                return 'HOT'
            elif total <= cold_thresh:
                return 'COLD'
            return 'UNIFORM'
        except ValueError as e:
            logger.error("Failed to classify board type: %s", e)
            raise

    def fuse_scores_vectorized(
        self, mod_scores: Dict[str, np.ndarray], board_type: str, default_weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Fuse multiple module scores using adaptive weights.

        Args:
            mod_scores (Dict[str, np.ndarray]): Module scores.
            board_type (str): Grid type ('HOT', 'COLD', 'UNIFORM').
            default_weights (Dict[str, float]): Default module weights.

        Returns:
            np.ndarray: Fused scores for hidden cells.

        Raises:
            ValueError: If mod_scores or weights are invalid.
        """
        try:
            if not mod_scores or not default_weights:
                raise ValueError("Module scores and default weights cannot be empty")

            w = self.weights_for(board_type, default_weights)
            names = list(mod_scores.keys())
            empty_yx = np.argwhere(list(mod_scores.values())[0] != 0)
            score_mat = np.stack([mod_scores[n][empty_yx[:, 0], empty_yx[:, 1]] for n in names], axis=1)
            weight_arr = np.array([w.get(n, 0.1) for n in names])
            heat_factor = np.abs(
                mod_scores.get('compute_dynamic_hot_cold_vectorized', np.zeros(score_mat.shape[0])).sum()
            ) / (score_mat.shape[0] + 1e-8)
            final = (score_mat.dot(weight_arr) / (weight_arr.sum() + 1e-8)) * (1 + heat_factor * 0.5)
            return np.where(final < 0.1, 0.1, final)
        except ValueError as e:
            logger.error("Failed to fuse scores: %s", e)
            raise

    def weights_for(self, board_type: str, default_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights based on grid type.

        Args:
            board_type (str): Grid type ('HOT', 'COLD', 'UNIFORM').
            default_weights (Dict[str, float]): Default weights.

        Returns:
            Dict[str, float]: Adjusted weights.

        Raises:
            ValueError: If board_type is invalid.
        """
        try:
            if board_type not in ['HOT', 'COLD', 'UNIFORM']:
                raise ValueError(f"Invalid board type: {board_type}")

            w = default_weights.copy()
            if board_type == 'HOT':
                w['compute_dynamic_hot_cold_advanced'] *= 1.5
                w['compute_block_heatmap_vectorized'] *= 1.2
            elif board_type == 'COLD':
                w['idw_vectorized'] *= 1.3
            else:
                w['analyze_number_patterns'] *= 1.2
                w['compute_difference_trend'] *= 1.1
            return w
        except ValueError as e:
            logger.error("Failed to adjust weights: %s", e)
            raise

    def predict_top3_vectorized(
        self, final_scores: np.ndarray, empty_positions: np.ndarray, target_num: Optional[int] = None
    ) -> List[Tuple[int, int, float, Dict[str, float]]]:
        """
        Predict top 3 positions for hidden numbers.

        Args:
            final_scores (np.ndarray): Final scores.
            empty_positions (np.ndarray): Hidden cell positions.
            target_num (Optional[int]): Target number.

        Returns:
            List[Tuple[int, int, float, Dict[str, float]]]: Top 3 prediction results.

        Raises:
            ValueError: If scores or positions are invalid.
        """
        try:
            if len(final_scores) != len(empty_positions):
                raise ValueError(f"Scores length {len(final_scores)} does not match positions {len(empty_positions)}")

            idxs = np.argsort(-final_scores)[:3]
            unique_idx = np.unique(idxs, return_index=True)[1]
            top3_idx = idxs[np.sort(unique_idx)[:3]]
            contributions = {
                name: float(final_scores[i]) for i, name in enumerate(self.MODULE_REGISTRY.keys()) if i in top3_idx
            }
            top3 = [
                (
                    int(empty_positions[i][0]),
                    int(empty_positions[i][1]),
                    max(float(final_scores[i]), 0.1),
                    contributions
                )
                for i in top3_idx
            ]
            return top3[:3]
        except ValueError as e:
            logger.error("Failed to predict top 3 positions: %s", e)
            raise

# Feature extraction for Faiss
def compute_global_features(heatmap: np.ndarray) -> np.ndarray:
    """
    Compute global heatmap statistics and gradient features for any grid size.

    Features include: mean, standard deviation, max, min, median, skew, kurtosis, gradient magnitude mean.

    Args:
        heatmap (np.ndarray): Heatmap array.

    Returns:
        np.ndarray: Vector of statistical and gradient features.

    Raises:
        ValueError: If heatmap is empty or invalid.
    """
    try:
        if heatmap.size == 0 or not np.isfinite(heatmap).all():
            raise ValueError("Heatmap array is empty or contains invalid values")

        # Statistical features
        stats = [
            float(heatmap.mean()),
            float(heatmap.std()),
            float(heatmap.max()),
            float(heatmap.min()),
            float(np.median(heatmap)),
            float(skew(heatmap.flatten())),
            float(kurtosis(heatmap.flatten()))
        ]

        # Gradient features
        grad_x = np.abs(np.diff(heatmap, axis=1)).mean() if heatmap.shape[1] > 1 else 0.0
        grad_y = np.abs(np.diff(heatmap, axis=0)).mean() if heatmap.shape[0] > 1 else 0.0
        grad_mean = float((grad_x + grad_y) / 2)
        stats.append(grad_mean)

        result = np.array(stats, dtype=np.float32)
        if not np.isfinite(result).all():
            logger.warning("Invalid global features computed, returning zeros")
            return np.zeros(8, dtype=np.float32)
        logger.debug("Computed global features: %s", result)
        return result
    except ValueError as e:
        logger.error("Failed to compute global features: %s", e)
        return np.zeros(8, dtype=np.float32)

def compute_local_patch(heatmap: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
    """
    Compute a local heatmap patch at the specified position and flatten it into a vector.

    Window size is dynamically chosen based on grid dimensions, suitable for small and large grids.

    Args:
        heatmap (np.ndarray): Heatmap array.
        pos (Tuple[int, int]): Target cell position (row, column).

    Returns:
        np.ndarray: Flattened local patch vector.

    Raises:
        ValueError: If heatmap shape or position is invalid.
        IndexError: If position is out of bounds.
    """
    try:
        if heatmap.ndim != 2:
            raise ValueError(f"Invalid heatmap shape {heatmap.shape}, expected 2D")
        M, N = heatmap.shape
        i, j = pos
        if not (0 <= i < M and 0 <= j < N):
            raise IndexError(f"Invalid position {pos} for heatmap shape {heatmap.shape}")

        # Dynamically choose window size
        window = min(M, N) // 2 * 2 + 1  # Ensure odd number
        window = max(3, min(window, min(M, N)))  # At least 3x3, max grid size
        pad = window // 2
        P = np.pad(heatmap, pad, mode="constant", constant_values=0)
        patch = P[i:i+window, j:j+window]
        result = patch.flatten().astype(np.float32)
        if not np.isfinite(result).all():
            logger.warning("Invalid local patch at %s, returning zeros", pos)
            return np.zeros(window * window, dtype=np.float32)
        logger.debug("Computed local patch at %s, shape %s", pos, result.shape)
        return result
    except (ValueError, IndexError) as e:
        logger.error("Failed to compute local patch: %s", e)
        return np.zeros(9, dtype=np.float32)  # Default 3x3 patch size

def compute_features(heatmap: np.ndarray, pos: Tuple[int, int]) -> np.ndarray:
    """
    Combine global statistical features and local patch features for full-board inference.

    Args:
        heatmap (np.ndarray): Heatmap array.
        pos (Tuple[int, int]): Target cell position (row, column).

    Returns:
        np.ndarray: Combined global and local feature vector.

    Raises:
        ValueError: If heatmap or position is invalid.
    """
    try:
        if heatmap.ndim != 2 or heatmap.size == 0:
            raise ValueError(f"Invalid heatmap shape {heatmap.shape} or empty")
        if not (0 <= pos[0] < heatmap.shape[0] and 0 <= pos[1] < heatmap.shape[1]):
            raise ValueError(f"Invalid position {pos} for heatmap shape {heatmap.shape}")

        g = compute_global_features(heatmap)
        l = compute_local_patch(heatmap, pos)
        features = np.concatenate([g, l])
        if not np.isfinite(features).all():
            logger.warning("Invalid combined features, returning zeros")
            return np.zeros(17, dtype=np.float32)  # 8 global + 9 local (3x3 default)
        logger.debug("Extracted features, shape %s, position %s, dimension %s", heatmap.shape, pos, features.shape)
        return features
    except ValueError as e:
        logger.error("Failed to compute features: %s", e)
        return np.zeros(17, dtype=np.float32)
