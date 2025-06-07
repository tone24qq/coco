# modules.py
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
import asyncio
import logging
from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScratchSolver:
    MODULE_REGISTRY = {}

    def __init__(self):
        self.tree = None
        self.known_yx = None
        self.known_vals = None
        self.MODULE_REGISTRY = {
            'compute_dynamic_hot_cold_vectorized': self.compute_dynamic_hot_cold_vectorized,
            'idw_vectorized': self.idw_vectorized,
            'compute_block_heatmap_vectorized': self.compute_block_heatmap_vectorized,
            'compute_global_diff_heatmap': self.compute_global_diff_heatmap,
            'compute_focus_score': self.compute_focus_score,
            'detect_skip_patterns': self.detect_skip_patterns,
            'compute_difference_trend': self.compute_difference_trend,
            'detect_mirror_sequences': self.detect_mirror_sequences,
            'connectivity_heatmap': self.connectivity_heatmap,
            'sequence_tail_analyzer': self.sequence_tail_analyzer
        }

    def update_tree(self, grid):
        self.known_yx = np.argwhere(grid != -1)
        self.known_vals = grid[grid != -1]
        if self.known_yx.size > 0:
            self.tree = cKDTree(self.known_yx)
        else:
            self.tree = None

    def idw_vectorized(self, grid):
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

    def compute_block_heatmap_vectorized(self, grid, block_size=3):
        """
        Vectorized block heatmap: each block mean - global mean.
        Output same shape as input.
        """
        arr = np.array(grid, dtype=float)
        windows = sliding_window_view(arr, (block_size, block_size))
        local_means = windows.mean(axis=(2,3))
        global_mean = arr.mean()
        heat = local_means - global_mean
        pad_h, pad_w = block_size - 1, block_size - 1
        heat_padded = np.pad(
            heat,
            ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
            mode='constant', constant_values=0
        )
        mn, mx = heat_padded.min(), heat_padded.max()
        return (heat_padded - mn) / (mx - mn + 1e-8)

    def compute_global_diff_heatmap(self, grid):
        """
        Second-order difference using Laplacian convolution.
        Output same shape as input.
        """
        arr = np.array(grid, dtype=float)
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=float)
        diff2 = convolve2d(arr, kernel, mode='same', boundary='symm')
        mn, mx = diff2.min(), diff2.max()
        return (diff2 - mn) / (mx - mn + 1e-8)

    def compute_focus_score(self, grid):
        mask = (grid != -1).astype(int)
        kernel = np.ones((3, 3)) / 9
        summed = convolve2d(np.where(grid != -1, grid, 0), kernel, mode='same', boundary='symm')
        count = convolve2d(mask, kernel, mode='same', boundary='symm')
        focus_map = summed / (count + 1e-8)
        scores = focus_map[grid == -1]
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=int)

    def detect_skip_patterns(self, grid):
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for k in range(1, min(4, M, N)):  # 支持 k=1~3
            for i in range(M):
                windows = sliding_window_view(grid[i], window_shape=k+1)
                for j in range(N - k):
                    if np.all(windows[j] != -1):
                        diff = np.diff(windows[j])
                        if np.all(np.abs(np.diff(diff)) < 1e-10):
                            step = diff[0]
                            for c in range(j+1, j+k):
                                if grid[i, c] == -1 and 1 <= grid[i, j] + step * (c - j) <= grid.max():
                                    scores[i, c] += 1.0 / k
                                    pred[i, c] = int(grid[i, j] + step * (c - j))
            for j in range(N):
                windows = sliding_window_view(grid[:, j], window_shape=k+1)
                for i in range(M - k):
                    if np.all(windows[i] != -1):
                        diff = np.diff(windows[i])
                        if np.all(np.abs(np.diff(diff)) < 1e-10):
                            step = diff[0]
                            for r in range(i+1, i+k):
                                if grid[r, j] == -1 and 1 <= grid[i, j] + step * (r - i) <= grid.max():
                                    scores[r, j] += 1.0 / k
                                    pred[r, j] = int(grid[i, j] + step * (r - i))
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        return scores[grid == -1], pred[grid == -1]

    def compute_difference_trend(self, grid):
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        d1 = np.diff(grid, axis=1)
        d2 = np.diff(grid, axis=0)
        diff_freq = np.bincount(d1.flatten(), minlength=grid.max()+1) + np.bincount(d2.flatten(), minlength=grid.max()+1)
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    if j >= 1 and grid[i, j-1] != -1:
                        expected = grid[i, j-1] + 1
                        if 1 <= expected <= grid.max() and diff_freq[1] > 0:
                            scores[i, j] = diff_freq[1] / diff_freq.sum()
                            pred[i, j] = int(expected)
                    if i >= 1 and grid[i-1, j] != -1:
                        expected = grid[i-1, j] + 1
                        if 1 <= expected <= grid.max() and diff_freq[1] > 0:
                            scores[i, j] = diff_freq[1] / diff_freq.sum()
                            pred[i, j] = int(expected)
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        return scores[grid == -1], pred[grid == -1]

    def detect_mirror_sequences(self, grid):
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        mid_x = N // 2
        mid_y = M // 2
        # 左右鏡像
        left = grid[:, :mid_x]
        right = np.fliplr(grid[:, N-mid_x:])[:, :mid_x]
        mirror_lr = np.all(left == right, axis=1)
        # 上下鏡像
        top = grid[:mid_y, :]
        bottom = np.flipud(grid[M-mid_y:, :])
        mirror_ud = np.all(top == bottom, axis=1)
        # 對角鏡像
        diag1 = grid.diagonal()
        diag2 = np.fliplr(grid).diagonal()
        mirror_diag = np.all(diag1 == diag2)
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    if j < mid_x and mirror_lr[i]:
                        scores[i, j] = 1.0
                        pred[i, j] = int(left[i, j % mid_x])
                    if i < mid_y and mirror_ud[j]:
                        scores[i, j] = 1.0
                        pred[i, j] = int(top[i % mid_y, j])
                    if mirror_diag and i == j:
                        scores[i, j] = 1.0
                        pred[i, j] = int(diag1[i])
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        return scores[grid == -1], pred[grid == -1]

    def connectivity_heatmap(self, grid):
        M, N = grid.shape
        mask = (grid != -1).astype(np.uint8)
        kernel_4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        kernel_8 = np.ones((3, 3)) - np.eye(3)
        conn_4 = convolve2d(mask, kernel_4, mode='same', boundary='symm')
        conn_8 = convolve2d(mask, kernel_8, mode='same', boundary='symm')
        conn_map = (conn_4 + conn_8) / 2
        scores = conn_map[grid == -1]
        return scores, np.full(np.count_nonzero(grid == -1), -1, dtype=int)

    def sequence_tail_analyzer(self, grid):
        M, N = grid.shape
        scores = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        tails = grid % 10
        freq = np.bincount(tails.flatten(), minlength=10) / (np.count_nonzero(grid != -1) + 1e-8)
        windows = sliding_window_view(grid, (3, 3))
        for i in range(M-2):
            for j in range(N-2):
                block = windows[i, j]
                block_tails = block[block != -1] % 10
                if block_tails.size > 0:
                    local_freq = np.bincount(block_tails, minlength=10) / (block_tails.size + 1e-8)
                    for y in range(i, i+3):
                        for x in range(j, j+3):
                            if grid[y, x] == -1:
                                best_tail = np.argmax(local_freq)
                                scores[y, x] = local_freq[best_tail]
                                candidates = grid[grid != -1][(grid[grid != -1] % 10) == best_tail]
                                if candidates.size > 0:
                                    pred[y, x] = int(min(candidates) + (best_tail * 10) if min(candidates) < 50 else -1)
        scores[grid != -1] = 0
        mn, mx = scores.min(), scores.max()
        scores = (scores - mn) / (mx - mn + 1e-8) if mx > mn else scores
        return scores[grid == -1], pred[grid == -1]

    def classify_board_type(self, dynamic_scores, hot_thresh=0.5, cold_thresh=-0.5):
        total = dynamic_scores.sum() / (np.count_nonzero(dynamic_scores != 0) + 1e-8)
        if total >= hot_thresh:
            return 'HOT'
        elif total <= cold_thresh:
            return 'COLD'
        else:
            return 'UNIFORM'

    def fuse_scores_vectorized(self, mod_scores, board_type, default_weights):
        w = self.weights_for(board_type, default_weights)
        names = list(mod_scores.keys())
        score_mat = np.stack([mod_scores[n] for n in names], axis=1)
        weight_arr = np.array([w.get(n, 0.1) for n in names])
        heat_factor = np.abs(mod_scores.get('compute_dynamic_hot_cold_vectorized', 0).sum()) / (np.count_nonzero(grid == -1) + 1e-8)
        final = (score_mat.dot(weight_arr) / (weight_arr.sum() + 1e-8)) * (1 + heat_factor * 0.5)
        return final

    def weights_for(self, board_type, default_weights):
        w = default_weights.copy()
        if board_type == 'HOT':
            w['compute_dynamic_hot_cold_vectorized'] *= 1.5
            w['compute_block_heatmap_vectorized'] *= 1.2
        elif board_type == 'COLD':
            w['idw_vectorized'] *= 1.3
        else:  # UNIFORM
            w['detect_mirror_sequences'] *= 1.2
            w['compute_difference_trend'] *= 1.1
        return w

    def predict_top3_vectorized(self, final_scores, empty_positions):
        idxs = np.argsort(-final_scores)[:3]
        unique_idx = np.unique(idxs, return_index=True)[1]
        top3_idx = idxs[np.sort(unique_idx)[:3]]
        contributions = {name: score for name, score in zip(self.MODULE_REGISTRY.keys(), final_scores)}
        return [(int(empty_positions[i][0]), int(empty_positions[i][1]), float(final_scores[i]), contributions) for i in top3_idx]