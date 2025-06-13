import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.signal import convolve2d
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
import pandas as pd

def compute_global_heatmap_from_files(files: List[str], batch_size: int = 1000):
    from brain import load_grid_from_file

    heatmap = initialize_heatmap()
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        for path in batch:
            grid = load_grid_from_file(path)
            update_heatmap(heatmap, grid)
        del batch, grid
    return heatmap

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
            'analyze_number_patterns': self.analyze_number_patterns,
            'compute_global_heatmap_from_files': self.compute_global_heatmap_from_files
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
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
        self.known_yx = np.argwhere(grid != -1)
        self.known_vals = grid[grid != -1]
        if self.known_yx.size > 0:
            self.tree = cKDTree(self.known_yx)
        else:
            self.tree = None
            self.known_yx = None
            self.known_vals = None

    def extract_multi_angle_features(self, grid: np.ndarray, output_path: str) -> Dict[str, Any]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

        def process_cell(i, j):
            if grid[i, j] == -1:
                return None
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

        results = Parallel(n_jobs=-1)(
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
        except OSError as e:
            logger.error(f"Failed to save features to {output_path}: {e}")

        return features_dict

    def idw_vectorized(self, grid: np.ndarray) -> np.ndarray:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
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
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def compute_dynamic_hot_cold_advanced(
        self, grid: np.ndarray, hot_q: float = 0.9, cold_q: float = 0.1, method: str = 'quantile'
    ) -> np.ndarray:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
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
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def compute_global_diff_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def compute_focus_score(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
        mask = (grid != -1).astype(np.int64)
        kernel = np.ones((3, 3)) / 9
        summed = convolve2d(np.where(grid != -1, grid, 0), kernel, mode='same', boundary='symm')
        count = convolve2d(mask, kernel, mode='same', boundary='symm')
        focus_map = summed / (count + 1e-8)
        scores = focus_map[grid == -1]
        scores = np.where(scores < 0.1, 0.1, scores)
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=np.int64)

    def detect_skip_patterns(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def compute_difference_trend(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def detect_mirror_sequences(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def connectivity_heatmap(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def sequence_tail_analyzer(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def analyze_number_patterns(self, grid: np.ndarray) -> Dict[Tuple[int, str], Dict[str, Any]]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
        M, N = grid.shape
        patterns: Dict[Tuple[int, str], Dict[str, Any]] = {}
        
        def find_arithmetic(arr: np.ndarray, min_len: int = 3) -> Optional[Dict[str, Any]]:
            if len(arr) < min_len:
                return None
            diffs = np.diff(arr)
            if np.all(np.abs(diffs - diffs[0]) < 1e-10):
                return {'type': 'arithmetic', 'diff': diffs[0]}
            return None
        
        def process_row(i):
            nums = grid[i][grid[i] != -1]
            if len(nums) >= 3:
                pattern = find_arithmetic(nums)
                if pattern:
                    return (i, 'h'), pattern
            return None
        
        def process_col(j):
            nums = grid[:, j][grid[:, j] != -1]
            if len(nums) >= 3:
                pattern = find_arithmetic(nums)
                if pattern:
                    return (j, 'v'), pattern
            return None
        
        row_results = Parallel(n_jobs=-1)(delayed(process_row)(i) for i in range(M))
        col_results = Parallel(n_jobs=-1)(delayed(process_col)(j) for j in range(N))
        
        for result in row_results + col_results:
            if result:
                patterns[result[0]] = result[1]
        
        return patterns

    def pattern_based_prediction(
        self, grid: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
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

    def heatmap_based_prediction(self, grid: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
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
        self, grid: np.ndarray, scores: np.ndarray, patterns: Dict[Tuple[int, str], Dict[str, Any]],
        global_heatmap_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        grid = grid.astype(np.int64)
        M, N = grid.shape
        predictions = np.full_like(grid, -1, dtype=float)
        confidence = np.zeros_like(grid, dtype=float)

        global_heatmap = np.zeros((M, N), dtype=float) if global_heatmap_path is None else \
                         self.load_global_heatmap(global_heatmap_path)[:M, :N]

        pattern_pred, pattern_scores = self.pattern_based_prediction(grid, patterns)
        heatmap_pred, heatmap_scores = self.heatmap_based_prediction(grid, scores)
        local_pred, local_scores = self.local_relationship_prediction(grid)
        
        w_pattern = 0.3
        w_heatmap = 0.3
        w_local = 0.2
        w_global = 0.2
        
        empty_mask = (grid == -1)
        empty_yx = np.where(empty_mask)
        for i, j in zip(empty_yx[0], empty_yx[1]):
            combined_pred = (
                pattern_pred[i, j] * w_pattern +
                heatmap_pred[i, j] * w_heatmap +
                local_pred[i, j] * w_local +
                global_heatmap[i, j] * w_global * (grid.size / (M * N))
            )
            predictions[i, j] = np.clip(combined_pred, 1, grid.size)

            confidence[i, j] = max(
                pattern_scores[i, j] * w_pattern,
                heatmap_scores[i, j] * w_heatmap,
                local_scores[i, j] * w_local,
                global_heatmap[i, j] * w_global
            )
            confidence[i, j] = max(confidence[i, j], 0.1)

        return predictions, confidence

    def evaluate_prediction(
        self, grid: np.ndarray, prediction: np.ndarray, true_values: np.ndarray
    ) -> Dict[str, float]:
        assert grid.ndim == 2, f"Expected 2D grid, got {grid.ndim}D array with shape {grid.shape}"
        assert prediction.ndim == 2, f"Expected 2D prediction, got {prediction.ndim}D array with shape {prediction.shape}"
        assert true_values.ndim == 2, f"Expected 2D true_values, got {true_values.ndim}D array with shape {true_values.shape}"
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

    def classify_board_type(self, dynamic_scores: np.ndarray, hot_thresh: float = 0.5, cold_thresh: float = -0.5) -> str:
        total = dynamic_scores.sum() / (np.count_nonzero(dynamic_scores != 0) + 1e-8)
        if total >= hot_thresh:
            return 'HOT'
        elif total <= cold_thresh:
            return 'COLD'
        return 'UNIFORM'

    def fuse_scores_vectorized(
        self, mod_scores: Dict[str, np.ndarray], board_type: str, default_weights: Dict[str, float]
    ) -> np.ndarray:
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

    def predict_top3_vectorized(
        self, final_scores: np.ndarray, empty_positions: np.ndarray, target_num: Optional[int] = None
    ) -> List[Tuple[int, int, float, Dict[str, float]]]:
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

    def load_global_heatmap(self, heatmap_path: str) -> np.ndarray:
        if not os.path.exists(heatmap_path):
            return np.zeros((20, 20), dtype=float)
        
        try:
            with open(heatmap_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            global_heatmap = np.array(data.get("global_heatmap", [[0.1] * 20] * 20), dtype=float)
            return global_heatmap
        except (OSError, json.JSONDecodeError, ValueError) as e:
            return np.zeros((20, 20), dtype=float)

    def compute_global_heatmap_from_files(self, files: List[str], batch_size: int = 1000, output_path: str = "samples/data/global_heatmap.json") -> Dict[str, Any]:
        global_heatmap = np.zeros((20, 20), dtype=float)
        total_cells = 0

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_heatmap = np.zeros((20, 20), dtype=float)
            batch_cells = 0

            for file_path in batch:
                try:
                    grids = load_grid_from_file(file_path)
                    for grid in grids:
                        m, n = grid.shape
                        padded_grid = np.pad(grid, ((0, 20 - m), (0, 20 - n)), mode='constant', constant_values=-1)
                        scores = self.compute_dynamic_hot_cold_vectorized(padded_grid)
                        empty_yx = np.argwhere(padded_grid == -1)
                        if len(scores) == len(empty_yx):
                            batch_heatmap[empty_yx[:, 0], empty_yx[:, 1]] += scores
                            batch_cells += len(empty_yx)
                except Exception as e:
                    logger.error(f"Failed to process grid from {file_path}: {e}")
                finally:
                    del grids

            if batch_cells > 0:
                global_heatmap += batch_heatmap
                total_cells += batch_cells

            del batch, batch_heatmap

        if total_cells > 0:
            global_heatmap = np.where(global_heatmap > 0, global_heatmap / total_cells, 0.1)
        else:
            global_heatmap = np.full((20, 20), 0.1)

        hot_spots = np.argwhere(global_heatmap > np.quantile(global_heatmap, 0.9))
        cold_spots = np.argwhere(global_heatmap < np.quantile(global_heatmap, 0.1))

        result = {
            "global_heatmap": global_heatmap.tolist(),
            "hot_spots": [{"row": int(r), "col": int(c), "score": float(global_heatmap[r, c])} for r, c in hot_spots],
            "cold_spots": [{"row": int(r), "col": int(c), "score": float(global_heatmap[r, c])} for r, c in cold_spots],
            "total_grids": len(files),
            "total_cells_analyzed": total_cells
        }

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to save global heatmap to {output_path}: {e}")

        return result

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
        except OSError as e:
            raise
    
    def load_history(self, filepath: str) -> None:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise