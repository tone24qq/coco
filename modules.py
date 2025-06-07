import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt
from ortools.sat.python import cp_model
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScratchSolver:
    def __init__(self):
        self.adaptive_weights = None

    def compute_focus_score(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        if np.all(grid == -1):
            score = np.ones((M, N), dtype=float) / (M * N)  # 均勻分數
            return score, np.full_like(grid, -1, dtype=int)
        mask = (grid != -1).astype(int)
        kernel = np.ones((3, 3), dtype=int)
        raw = convolve2d(mask, kernel, mode='same', boundary='fill', fillvalue=0)
        score = np.zeros_like(raw, dtype=float)
        pred = np.full_like(grid, -1, dtype=int)
        score[grid == -1] = raw[grid == -1]
        if np.any(score):
            mn, mx = score.min(), score.max()
            score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        else:
            score[grid == -1] = 1.0 / np.sum(grid == -1)  # 均勻分數
        return score, pred

    def detect_skip_patterns(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.sum(grid != -1) < 1:  # 至少一個已知格
            score[grid == -1] = 1.0 / np.sum(grid == -1)
            return score, pred
        for i in range(M):
            known_cols = np.where(grid[i] != -1)[0]
            if len(known_cols) < 1:
                continue
            for idx in range(len(known_cols)):
                c = known_cols[idx]
                v = grid[i, c]
                for c_next in range(N):
                    if grid[i, c_next] == -1:
                        expected = v  # 單格推測
                        if 1 <= expected <= grid.max():
                            score[i, c_next] += 0.5  # 降低單格信心
                            pred[i, c_next] = int(expected)
        for j in range(N):
            known_rows = np.where(grid[:, j] != -1)[0]
            if len(known_rows) < 1:
                continue
            for idx in range(len(known_rows)):
                r = known_rows[idx]
                v = grid[r, j]
                for r_next in range(M):
                    if grid[r_next, j] == -1:
                        expected = v
                        if 1 <= expected <= grid.max():
                            score[r_next, j] += 0.5
                            pred[r_next, j] = int(expected)
        if np.any(score):
            mn, mx = score.min(), score.max()
            score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        else:
            score[grid == -1] = 1.0 / np.sum(grid == -1)
        return score, pred

    def compute_difference_trend(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.sum(grid != -1) < 1:
            score[grid == -1] = 1.0 / np.sum(grid == -1)
            return score, pred
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    continue
                if j >= 1 and grid[i, j-1] != -1:
                    expected = grid[i, j-1]
                    if 1 <= expected <= grid.max():
                        score[i, j] = 0.5
                        pred[i, j] = int(expected)
                if i >= 1 and grid[i-1, j] != -1:
                    expected = grid[i-1, j]
                    if 1 <= expected <= grid.max():
                        score[i, j] = 0.5
                        pred[i, j] = int(expected)
        if np.any(score):
            mn, mx = score.min(), score.max()
            score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        else:
            score[grid == -1] = 1.0 / np.sum(grid == -1)
        return score, pred

    def detect_mirror_sequences(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.sum(grid != -1) < 1:
            score[grid == -1] = 1.0 / np.sum(grid == -1)
            return score, pred
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    continue
                max_k = min(j, N - j - 1)
                for k in range(1, max_k + 1):
                    left = grid[i, j - k]
                    right = grid[i, j + k]
                    if left != -1 and right != -1 and abs(left - right) < 1e-10:
                        score[i, j] = 0.5
                        pred[i, j] = int(left)
                        break
                if score[i, j] == 0:
                    max_k2 = min(i, M - i - 1)
                    for k in range(1, max_k2 + 1):
                        up = grid[i - k, j]
                        down = grid[i + k, j]
                        if up != -1 and down != -1 and abs(up - down) < 1e-10:
                            score[i, j] = 0.5
                            pred[i, j] = int(up)
                            break
        if np.any(score):
            mn, mx = score.min(), score.max()
            score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        else:
            score[grid == -1] = 1.0 / np.sum(grid == -1)
        return score, pred