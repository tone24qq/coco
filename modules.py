import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt
from sklearn.neighbors import KDTree

class ScratchSolver:
    def __init__(self):
        pass

    def compute_focus_score(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.all(grid == -1):
            return np.zeros_like(grid, dtype=float), np.full_like(grid, -1, dtype=int)
        mask = (grid != -1).astype(int)
        kernel = np.ones((3, 3), dtype=int)
        raw = convolve2d(mask, kernel, mode='same', boundary='fill', fillvalue=0)
        score = np.zeros_like(raw, dtype=float)
        pred = np.full_like(grid, -1, dtype=int)
        score[grid == -1] = raw[grid == -1]
        mn, mx = score.min(), score.max()
        if mx > mn:
            score = (score - mn) / (mx - mn)
        return score, pred

    def detect_skip_patterns(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for i in range(M):
            known_cols = np.where(grid[i] != -1)[0]
            if len(known_cols) < 2:
                continue
            for idx in range(len(known_cols)-1):
                c1, c2 = known_cols[idx], known_cols[idx+1]
                v1, v2 = grid[i, c1], grid[i, c2]
                diff = v2 - v1
                step = diff // (c2 - c1) if (c2 - c1) != 0 else 0
                if (c2 - c1) * step != diff:
                    continue
                for c in range(c1+1, c2):
                    expected = v1 + step * (c - c1)
                    if grid[i, c] == -1 and 1 <= expected <= grid.max():
                        score[i, c] += 1.0
                        pred[i, c] = expected
        for j in range(N):
            known_rows = np.where(grid[:, j] != -1)[0]
            if len(known_rows) < 2:
                continue
            for idx in range(len(known_rows)-1):
                r1, r2 = known_rows[idx], known_rows[idx+1]
                v1, v2 = grid[r1, j], grid[r2, j]
                diff = v2 - v1
                step = diff // (r2 - r1) if (r2 - r1) != 0 else 0
                if (r2 - r1) * step != diff:
                    continue
                for r in range(r1+1, r2):
                    expected = v1 + step * (r - r1)
                    if grid[r, j] == -1 and 1 <= expected <= grid.max():
                        score[r, j] += 1.0
                        pred[r, j] = expected
        mn, mx = score.min(), score.max()
        if mx > mn:
            score = (score - mn) / (mx - mn)
        return score, pred

    def compute_difference_trend(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    continue
                if j >= 2 and grid[i, j-1] != -1 and grid[i, j-2] != -1:
                    d = grid[i, j-1] - grid[i, j-2]
                    expected = grid[i, j-1] + d
                    if 1 <= expected <= grid.max():
                        score[i, j] = 1.0
                        pred[i, j] = expected
                if i >= 2 and grid[i-1, j] != -1 and grid[i-2, j] != -1:
                    d = grid[i-1, j] - grid[i-2, j]
                    expected = grid[i-1, j] + d
                    if 1 <= expected <= grid.max():
                        score[i, j] = 1.0
                        pred[i, j] = expected
        return score, pred

    def detect_mirror_sequences(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    continue
                max_k = min(j, N - j - 1)
                for k in range(1, max_k + 1):
                    left = grid[i, j - k]
                    right = grid[i, j + k]
                    if left != -1 and right != -1 and left == right:
                        score[i, j] = 1.0
                        pred[i, j] = left
                        break
                if score[i, j] == 1:
                    continue
                max_k2 = min(i, M - i - 1)
                for k in range(1, max_k2 + 1):
                    up = grid[i - k, j]
                    down = grid[i + k, j]
                    if up != -1 and down != -1 and up == down:
                        score[i, j] = 1.0
                        pred[i, j] = up
                        break
        return score, pred

    def connectivity_heatmap(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.all(grid == -1):
            return score, pred
        mask = (grid != -1).astype(np.uint8)
        dist_map = distance_transform_edt(1 - mask)
        score[grid == -1] = 1.0 / (dist_map[grid == -1] + 1e-6)
        mn, mx = score.min(), score.max()
        if mx > mn:
            score = (score - mn) / (mx - mn)
        return score, pred

    def sequence_tail_analyzer(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.all(grid == -1):
            return score, pred
        known_positions = np.argwhere(grid != -1)
        tail_positions = {t: [] for t in range(10)}
        for x, y in known_positions:
            t = int(grid[x, y] % 10)
            tail_positions[t].append((x, y))
        tail_counts = {t: len(pos) for t, pos in tail_positions.items()}
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    continue
                best_score, best_tail = 0.0, -1
                for t in range(10):
                    positions_t = tail_positions[t]
                    if not positions_t:
                        continue
                    coords = np.array(positions_t)
                    dists = np.abs(coords - np.array([i, j])).sum(axis=1)
                    min_dist = np.min(dists)
                    s = tail_counts[t] / (min_dist + 1e-6)
                    if s > best_score:
                        best_score, best_tail = s, t
                score[i, j] = best_score
                if best_tail >= 0:
                    candidates = [v for x, y in known_positions if int(grid[x, y] % 10) == best_tail]
                    if candidates:
                        pred[i, j] = min(candidates) + (best_tail * 10) if min(candidates) < 50 else -1
        return score, pred

    def fuse_scores(self, gridscores: dict, grid: np.ndarray, gridpreds: dict) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        final_score = np.zeros((M, N), dtype=float)
        final_pred = np.full((M, N), -1, dtype=int)
        weights = gridscores.get('_weights', {})
        for name, score_arr in gridscores.items():
            if name == '_weights':
                continue
            w = weights.get(name, 0.0)
            final_score += w * score_arr
        final_score[grid != -1] = 0
        mn, mx = final_score.min(), final_score.max()
        if mx > mn:
            final_score = (final_score - mn) / (mx - mn)
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    final_pred[i, j] = grid[i, j]
                    continue
                numerator = 0.0
                denominator = 0.0
                for name, pred_arr in gridpreds.items():
                    score_arr = gridscores[name]
                    w = weights.get(name, 0.0)
                    s = score_arr[i, j]
                    if s > 0 and pred_arr[i, j] >= 0:
                        numerator += pred_arr[i, j] * w * s
                        denominator += w * s
                if denominator > 0:
                    final_pred[i, j] = int(round(numerator / (denominator + 1e-6)))
                else:
                    final_pred[i, j] = -1
        return final_score, final_pred