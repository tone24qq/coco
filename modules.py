import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt
from ortools.sat.python import cp_model
from sklearn.linear_model import LinearRegression

class ScratchSolver:
    def __init__(self):
        self.adaptive_weights = None

    def compute_focus_score(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.all(grid == -1):
            score = np.ones_like(grid, dtype=float) / grid.size  # 空白卡均勻分數
            return score, np.full_like(grid, -1, dtype=int)
        mask = (grid != -1).astype(int)
        kernel = np.ones((3, 3), dtype=int)
        raw = convolve2d(mask, kernel, mode='same', boundary='fill', fillvalue=0)
        score = np.zeros_like(raw, dtype=float)
        pred = np.full_like(grid, -1, dtype=int)
        score[grid == -1] = raw[grid == -1]
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score, pred

    def detect_skip_patterns(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for i in range(M):
            known_cols = np.where(grid[i] != -1)[0]
            if len(known_cols) < 2:
                if len(known_cols) == 1:  # 單格推測
                    score[i, grid[i] == -1] = 0.1  # 低分數
                continue
            for idx in range(len(known_cols)-1):
                c1, c2 = known_cols[idx], known_cols[idx+1]
                v1, v2 = grid[i, c1], grid[i, c2]
                diff = v2 - v1
                step = diff / (c2 - c1) if (c2 - c1) != 0 else 0
                if abs((c2 - c1) * step - diff) < 1e-10:
                    for c in range(c1+1, c2):
                        expected = v1 + step * (c - c1)
                        if grid[i, c] == -1 and 1 <= expected <= grid.max():
                            score[i, c] += 1.0
                            pred[i, c] = int(round(expected))
        for j in range(N):
            known_rows = np.where(grid[:, j] != -1)[0]
            if len(known_rows) < 2:
                if len(known_rows) == 1:
                    score[grid[:, j] == -1, j] = 0.1
                continue
            for idx in range(len(known_rows)-1):
                r1, r2 = known_rows[idx], known_rows[idx+1]
                v1, v2 = grid[r1, j], grid[r2, j]
                diff = v2 - v1
                step = diff / (r2 - r1) if (r2 - r1) != 0 else 0
                if abs((r2 - r1) * step - diff) < 1e-10:
                    for r in range(r1+1, r2):
                        expected = v1 + step * (r - r1)
                        if grid[r, j] == -1 and 1 <= expected <= grid.max():
                            score[r, j] += 1.0
                            pred[r, j] = int(round(expected))
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score, pred

    def compute_difference_trend(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    continue
                if j >= 1 and grid[i, j-1] != -1:  # 放寬至單格
                    expected = grid[i, j-1] + 1  # 簡單假設
                    if 1 <= expected <= grid.max():
                        score[i, j] = 0.5
                        pred[i, j] = int(expected)
                if i >= 1 and grid[i-1, j] != -1:
                    expected = grid[i-1, j] + 1
                    if 1 <= expected <= grid.max():
                        score[i, j] = 0.5
                        pred[i, j] = int(expected)
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
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
                    if left != -1 and right != -1 and abs(left - right) < 1e-10:
                        score[i, j] = 1.0
                        pred[i, j] = int(left)
                        break
                if score[i, j] == 1:
                    continue
                max_k2 = min(i, M - i - 1)
                for k in range(1, max_k2 + 1):
                    up = grid[i - k, j]
                    down = grid[i + k, j]
                    if up != -1 and down != -1 and abs(up - down) < 1e-10:
                        score[i, j] = 1.0
                        pred[i, j] = int(up)
                        break
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score, pred

    def connectivity_heatmap(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.all(grid == -1):
            score = np.ones_like(grid, dtype=float) / grid.size
            return score, pred
        mask = (grid != -1).astype(np.uint8)
        dist_map = distance_transform_edt(1 - mask)
        score[grid == -1] = 1.0 / (dist_map[grid == -1] + 1e-6)
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score, pred

    def sequence_tail_analyzer(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        pred = np.full((M, N), -1, dtype=int)
        if np.all(grid == -1):
            score = np.ones_like(grid, dtype=float) / grid.size
            return score, pred
        known_positions = np.argwhere(grid != -1)
        if len(known_positions) == 0:
            score[grid == -1] = 0.1
            return score, pred
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
                    candidates = [grid[x, y] for x, y in known_positions if int(grid[x, y] % 10) == best_tail]
                    if candidates:
                        pred[i, j] = min(candidates) + (best_tail * 10) if min(candidates) < 50 else -1
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score, pred

    def constraint_solver(self, grid: np.ndarray, target_num: int) -> np.ndarray:
        if target_num is None or target_num not in range(1, grid.max() + 1):
            return np.ones_like(grid, dtype=float) / grid.size
        M, N = grid.shape
        model = cp_model.CpModel()
        vars = {}
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    vars[i, j] = model.NewIntVar(1, grid.max(), f'cell_{i}_{j}')
        for i in range(M):
            model.AddAllDifferent([vars[i, j] for j in range(N) if (i, j) in vars])
        for j in range(N):
            model.AddAllDifferent([vars[i, j] for i in range(M) if (i, j) in vars])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2.0  # 放寬時間
        status = solver.Solve(model)
        score = np.zeros((M, N), dtype=float)
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            for (i, j), v in vars.items():
                if solver.Value(v) == target_num:
                    score[i, j] = 1.0
        else:
            score[grid == -1] = 1.0 / np.sum(grid == -1)  # 均勻分數
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score

    def tensor_full_score(self, grid: np.ndarray) -> np.ndarray:
        M, N = grid.shape
        tensor = np.zeros((M, N, 2), dtype=float)
        tensor[:, :, 0] = (grid != -1).astype(float)
        kernel = np.ones((3, 3), dtype=float) / 9
        tensor[:, :, 1] = convolve2d((grid != -1).astype(float), kernel, mode='same', boundary='symm')
        conv_score = convolve2d(tensor[:, :, 1], kernel, mode='same', boundary='symm')
        score = np.zeros((M, N), dtype=float)
        score[grid == -1] = conv_score[grid == -1]
        if np.all(score == 0):
            score[grid == -1] = 0.1
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score

    def pattern_mining(self, grid: np.ndarray) -> np.ndarray:
        M, N = grid.shape
        score = np.zeros((M, N), dtype=float)
        known_nums = grid[grid != -1]
        if len(known_nums) == 0:
            score[grid == -1] = 0.1
            return score
        freq = np.histogram(known_nums % 10, bins=range(11))[0] / (len(known_nums) + 1e-6)
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    score[i, j] = np.mean(freq)
        mn, mx = score.min(), score.max()
        score = (score - mn) / (mx - mn + 1e-10) if mx > mn else score
        return score

    def dynamic_weights(self, grid, scores, initial_weights, json_scores=None):
        weights = initial_weights.copy()
        total_weight = sum(weights.values()) + 1e-6
        weights = {k: v / total_weight for k, v in weights.items()}
        contributions = {k: np.mean(s[grid == -1]) if np.any(s[grid == -1]) else 0.1 
                        for k, s in scores.items() if k != '_weights'}
        if json_scores is not None and np.any(json_scores):
            contributions['json'] = np.mean(json_scores[grid == -1])
        total_contrib = sum(contributions.values()) + 1e-6
        if total_contrib > 0:
            for k in weights:
                weights[k] *= contributions.get(k, 0.1) / total_contrib
            total = sum(weights.values()) + 1e-6
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def fuse_scores(self, gridscores: dict, grid: np.ndarray, gridpreds: dict, target_num: int) -> tuple[np.ndarray, np.ndarray]:
        M, N = grid.shape
        final_score = np.zeros((M, N), dtype=float)
        final_pred = np.full((M, N), -1, dtype=int)
        weights = gridscores.get('_weights', {})
        for name, score_arr in gridscores.items():
            if name == '_weights':
                continue
            w = weights.get(name, 0.1)
            final_score += w * score_arr
        final_score[grid != -1] = 0
        mn, mx = final_score.min(), final_score.max()
        final_score = (final_score - mn) / (mx - mn + 1e-10) if mx > mn else final_score
        for i in range(M):
            for j in range(N):
                if grid[i, j] != -1:
                    final_pred[i, j] = grid[i, j]
                    continue
                numerator = 0.0
                denominator = 0.0
                for name, pred_arr in gridpreds.items():
                    score_arr = gridscores.get(name, np.zeros_like(grid))
                    w = weights.get(name, 0.1)
                    s = score_arr[i, j]
                    if s > 0 and pred_arr[i, j] >= 0:
                        numerator += pred_arr[i, j] * w * s
                        denominator += w * s
                if denominator > 0:
                    final_pred[i, j] = int(round(numerator / (denominator + 1e-6)))
                else:
                    final_pred[i, j] = -1
        return final_score, final_pred

    def predict_specific_number(self, grid, final_score, target_num, weights):
        M, N = grid.shape
        if target_num in grid[grid != -1]:
            return None
        candidates = []
        for i in range(M):
            for j in range(N):
                if grid[i, j] == -1:
                    score = final_score[i, j]
                    candidates.append((i, j, score, self._reasoning(i, j, target_num, weights)))
        if not candidates:
            return None
        best = max(candidates, key=lambda x: x[2])
        return (best[0], best[1], best[2], best[3])

    def default_candidate(self, grid, target_num, weights):
        M, N = grid.shape
        candidates = [(i, j, 1.0 / np.sum(grid == -1), "均勻分配") 
                      for i in range(M) for j in range(N) if grid[i, j] == -1]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[2])

    def _reasoning(self, i, j, target_num, weights):
        reasoning = []
        for module, score in [('focus', weights['focus']), ('skip', weights['skip']), ('diff', weights['diff']),
                             ('mirror', weights['mirror']), ('conn', weights['conn']), ('tail', weights['tail']),
                             ('constraint', weights['constraint']), ('tensor', weights['tensor']),
                             ('pattern', weights['pattern'])]:
            if score > 0.05:  # 降低門檻
                reasoning.append(f"{module}: {score:.2f}")
        return "; ".join(reasoning) if reasoning else "均勻分配"