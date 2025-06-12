# modules.py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
import logging
import json
import os
from typing import Dict, List, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScratchSolver:
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return
        self.known_yx = np.argwhere(grid != -1)
        self.known_vals = grid[grid != -1]
        if self.known_yx.size > 0:
            self.tree = cKDTree(self.known_yx)
        else:
            self.tree = None

    def extract_multi_angle_features(self, grid: np.ndarray, output_path: str) -> Dict[str, Any]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return {}
        M, N = grid.shape
        features_dict: Dict[str, Any] = {
            "row_features": {},
            "col_features": {},
            "diagonal_features": {},
            "neighborhood_features": {},
            "difference_features": {}
        }

        for i in range(M):
            for j in range(N):
                num = grid[i, j]
                if num == -1:
                    continue
                features_dict["row_features"].setdefault(i, []).append(num)
                features_dict["col_features"].setdefault(j, []).append(num)
                if i == j:
                    features_dict["diagonal_features"].setdefault("main", []).append(num)
                if i + j == M - 1:
                    features_dict["diagonal_features"].setdefault("anti", []).append(num)
                window = sliding_window_view(
                    np.pad(grid, ((1, 1), (1, 1)), mode='edge'), (3, 3)
                )[i, j]
                neighbors = window[window != -1].flatten()
                features_dict["neighborhood_features"].setdefault(f"{i},{j}", []).extend(neighbors.tolist())
                diffs = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < M and 0 <= nj < N and grid[ni, nj] != -1:
                        diffs.append(abs(num - grid[ni, nj]))
                features_dict["difference_features"].setdefault(f"{i},{j}", diffs)

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(features_dict, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

        return features_dict

    def idw_vectorized(self, grid: np.ndarray) -> np.ndarray:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.array([])
        empty_yx = np.argwhere(grid == -1)
        if empty_yx.size == 0:
            return np.array([])
        if self.tree is None or self.known_yx is None or self.known_vals is None:
            return np.full(empty_yx.shape[0], 0.1)
        dists, idxs = self.tree.query(empty_yx, k=min(5, self.known_yx.shape[0]))
        weights = 1.0 / (dists ** 2 + 1e-8)
        est = np.sum(weights * self.known_vals[idxs], axis=1) / np.sum(weights, axis=1)
        return np.where(est < 0.1, 0.1, est)

    def compute_dynamic_hot_cold_vectorized(self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'quantile') -> np.ndarray:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.array([])
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

    def compute_dynamic_hot_cold_advanced(self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'quantile') -> np.ndarray:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.array([])
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.array([])
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
        mask = (grid != -1).astype(int)
        kernel = np.ones((3, 3)) / 9
        summed = convolve2d(np.where(grid != -1, grid, 0), kernel, mode='same', boundary='symm')
        count = convolve2d(mask, kernel, mode='same', boundary='symm')
        focus_map = summed / (count + 1e-8)
        scores = focus_map[grid == -1]
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=int)

    def detect_skip_patterns(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        d1 = np.diff(grid, axis=1)
        d2 = np.diff(grid, axis=0)
        diff_freq = np.bincount(d1.flatten()[d1.flatten() != -1].astype(int), minlength=grid.size+1) + \
                    np.bincount(d2.flatten()[d2.flatten() != -1].astype(int), minlength=grid.size+1)
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
        M, N = grid.shape
        if M < 3 or N < 3 or np.all(grid == -1):
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)

        try:
            padded = np.pad(grid, ((1, 1), (1, 1)), mode='edge')
            windows = sliding_window_view(padded, (3, 3))
            if windows.shape[0] != M or windows.shape[1] != N:
                raise ValueError
        except Exception:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)

        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        mid_x, mid_y = N // 2, M // 2

        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    try:
                        if j < mid_x and N - 1 - j < N and grid[i, N - 1 - j] != -1:
                            scores[i, j] = 1.0
                            pred[i, j] = int(grid[i, N - 1 - j])
                        if i < mid_y and M - 1 - i < M and grid[M - 1 - i, j] != -1:
                            scores[i, j] = 1.0
                            pred[i, j] = int(grid[M - 1 - i, j])
                        if i == j and M - 1 - i < M and grid[M - 1 - i, M - 1 - i] != -1:
                            scores[i, j] = 1.0
                            pred[i, j] = int(grid[M - 1 - i, M - 1 - i])
                    except IndexError:
                        continue

        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores[grid == -1], pred[grid == -1]

    def connectivity_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
        M, N = grid.shape
        mask = (grid != -1).astype(np.uint8)
        kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        kernel_8 = np.ones((3, 3)) - np.eye(3)
        conn_4 = convolve2d(mask, kernel_4, mode='same', boundary='symm')
        conn_8 = convolve2d(mask, kernel_8, mode='same', boundary='symm')
        conn_map = (conn_4 + conn_8) / 2
        scores = conn_map[grid == -1]
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=int)

    def sequence_tail_analyzer(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
        M, N = grid.shape
        if M < 1 or N < 1 or np.all(grid == -1):
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)

        try:
            padded = np.pad(grid, ((1, 1), (1, 1)), mode='edge')
            windows = sliding_window_view(padded, (3, 3))
        except Exception:
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)

        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        known_tails = grid[grid != -1] % 10
        freq = np.bincount(known_tails, minlength=10) / (np.count_nonzero(grid != -1) + 1e-8)

        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    window = windows[i, j]
                    block_tails = window[window != -1] % 10
                    if block_tails.size > 0:
                        local_freq = np.bincount(block_tails, minlength=10) / (block_tails.size + 1e-8)
                        best_tail = np.argmax(local_freq)
                        scores[i, j] = local_freq[best_tail]
                        candidates = grid[grid != -1][(grid[grid != -1] % 10) == best_tail]
                        if candidates.size > 0:
                            base_value = np.min(candidates)
                            pred[i, j] = int(base_value + (best_tail * 10)) if base_value < 50 else -1

        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores[grid == -1], pred[grid == -1]

    def analyze_number_patterns(self, grid: np.ndarray) -> Dict[Tuple[int, str], Dict[str, Any]]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return {}
        M, N = grid.shape
        patterns: Dict[Tuple[int, str], Dict[str, Any]] = {}
        
        def find_arithmetic(arr: np.ndarray, min_len: int = 3) -> Optional[Dict[str, Any]]:
            if len(arr) < min_len:
                return None
            diffs = np.diff(arr)
            if np.all(np.abs(diffs - diffs[0]) < 1e-10):
                return {'type': 'arithmetic', 'diff': diffs[0]}
            return None
        
        for i in range(M):
            nums = grid[i][grid[i] != -1]
            if len(nums) >= 3:
                pattern = find_arithmetic(nums)
                if pattern:
                    patterns[(i, 'h')] = pattern
        
        for j in range(N):
            nums = grid[:, j][grid[:, j] != -1]
            if len(nums) >= 3:
                pattern = find_arithmetic(nums)
                if pattern:
                    patterns[(j, 'v')] = pattern
        
        return patterns

    def pattern_based_prediction(self, grid: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
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
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
        pred = np.zeros_like(grid, dtype=float)
        confidence = np.zeros_like(grid, dtype=float)
        empty_yx = np.argwhere(grid == -1)
        if len(scores) == len(empty_yx):
            pred[empty_yx[:, 0], empty_yx[:, 1]] = scores
            confidence[empty_yx[:, 0], empty_yx[:, 1]] = scores
        pred = np.clip(pred, 1, grid.size)
        confidence = np.where(confidence < 0.1, 0.1, confidence)
        return pred, confidence

    def integrate_predictions(self, grid: np.ndarray, scores: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
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

    def evaluate_prediction(self, grid: np.ndarray, prediction: np.ndarray, true_values: np.ndarray) -> Dict[str, float]:
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return {'accuracy': 0.0, 'pattern_match': 0.0, 'value_diff': 0.0}
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

    def classify_board_type(self, dynamic_scores: np.ndarray, hot_thresh: float = 0.5, cold_thresh: float = -0.5) -> str:
        if not isinstance(dynamic_scores, np.ndarray) or dynamic_scores.size == 0:
            return 'UNIFORM'
        total = dynamic_scores.sum() / (np.count_nonzero(dynamic_scores != 0) + 1e-8)
        if total >= hot_thresh:
            return 'HOT'
        elif total <= cold_thresh:
            return 'COLD'
        return 'UNIFORM'

    def fuse_scores_vectorized(self, mod_scores: Dict[str, np.ndarray], board_type: str, default_weights: Dict[str, float]) -> np.ndarray:
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

    def weights_for(self, board_type: str, default_weights: Dict[str, float]) -> Dict[str, float]:
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

    def predict_top3_vectorized(self, final_scores: np.ndarray, empty_positions: np.ndarray, target_num: Optional[int] = None) -> List[Tuple[int, int, float, Dict[str, float]]]:
        if not isinstance(final_scores, np.ndarray) or final_scores.size == 0 or empty_positions.size == 0:
            return []
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

class AdaptiveWeights:
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights.copy()
        self.history: List[Dict[str, Any]] = []
    
    def update(self, success_rate: float, module_scores: Dict[str, np.ndarray]) -> None:
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
    
    def save_history(self, filepath: str) -> None:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
    
    def load_history(self, filepath: str) -> None:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass