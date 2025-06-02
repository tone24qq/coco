# new_module.py

import math
import random

class ProbabilityModule:
    """
    對所有隱藏格均勻分配概率（基線）。
    """
    def predict(self, grid: list[list[int]], target: int) -> dict[int, float]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        hidden = [
            i * cols + j + 1
            for i in range(rows)
            for j in range(cols)
            if grid[i][j] == -1
        ]
        if not hidden:
            return {}
        prob = 1.0 / len(hidden)
        return {pos: prob for pos in hidden}


class AdjacencyModule:
    """
    根據鄰近已揭示數字與 target 的差距打分。
    差距越小，分數越高；若都沒鄰近貢獻，則將所有隱藏格設為 1。
    """
    def predict(self, grid: list[list[int]], target: int) -> dict[int, float]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        scores: dict[int, float] = {}
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == -1:
                    pos_id = i * cols + j + 1
                    score = 0.0
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != -1:
                            diff = abs(grid[ni][nj] - target)
                            score += 1.0 / (diff + 1.0)
                    scores[pos_id] = score

        if scores and all(v == 0 for v in scores.values()):
            for k in scores:
                scores[k] = 1.0

        return scores


class FrequencyModule:
    """
    將所有數字分「低 ≤ N/2」與「高 > N/2」，統計四象限內已揭示目標類別數量，
    給被選中（最少已揭示）的象限內隱藏格權重 2，其餘為 1。
    """
    def predict(self, grid: list[list[int]], target: int) -> dict[int, float]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        N = rows * cols
        if N == 0:
            return {}

        half = N / 2.0
        target_cat = 'low' if target <= half else 'high'

        mid_row = rows // 2
        mid_col = cols // 2
        quad_count = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        quad_hidden: dict[str, list[int]] = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

        for i in range(rows):
            for j in range(cols):
                if i < mid_row:
                    q = 'Q1' if j < mid_col else 'Q2'
                else:
                    q = 'Q3' if j < mid_col else 'Q4'

                if grid[i][j] == -1:
                    quad_hidden[q].append(i * cols + j + 1)
                else:
                    val_cat = 'low' if grid[i][j] <= half else 'high'
                    if val_cat == target_cat:
                        quad_count[q] += 1

        min_cnt = min(quad_count.values())
        candidates = [q for q, cnt in quad_count.items() if cnt == min_cnt]

        scores: dict[int, float] = {}
        for q, hidden_list in quad_hidden.items():
            base = 2.0 if q in candidates else 1.0
            for pos in hidden_list:
                scores[pos] = base

        return scores


class PatternModule:
    """
    深度檢測「行範圍模式」或「列餘數模式」：
      1. 行範圍模式：假設每行是連續號段 (e.g. 1–10、11–20…)。
      2. 列餘數模式：假設每列數字末位相同 (j 列末位 = j+1，最後一列末位 = 0)。
    若皆不符，則套用「半區啟發式」：target > N/2 偏好下半，否則偏好上半。
    """
    def predict(self, grid: list[list[int]], target: int) -> dict[int, float]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            return {}

        range_size = math.ceil((rows * cols) / rows)

        # 檢測行範圍模式
        row_pattern = True
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != -1:
                    val = grid[i][j]
                    start = i * range_size + 1
                    end = (i + 1) * range_size
                    if not (start <= val <= end):
                        row_pattern = False
                        break
            if not row_pattern:
                break

        # 檢測列餘數模式
        col_pattern = True
        for j in range(cols):
            for i in range(rows):
                if grid[i][j] != -1:
                    val = grid[i][j]
                    last_digit = val % 10
                    expected = 0 if j == cols - 1 else (j + 1)
                    if last_digit != expected:
                        col_pattern = False
                        break
            if not col_pattern:
                break

        scores: dict[int, float] = {}
        if row_pattern:
            target_row = (target - 1) // range_size
            if 0 <= target_row < rows:
                for j in range(cols):
                    if grid[target_row][j] == -1:
                        pos_id = target_row * cols + j + 1
                        scores[pos_id] = 1.0

        elif col_pattern:
            last_digit = target % 10
            target_col = (last_digit - 1) if last_digit != 0 else (cols - 1)
            if 0 <= target_col < cols:
                for i in range(rows):
                    if grid[i][target_col] == -1:
                        pos_id = i * cols + target_col + 1
                        scores[pos_id] = 1.0

        else:
            mid = (rows * cols) / 2.0
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == -1:
                        pos_id = i * cols + j + 1
                        if target > mid:
                            scores[pos_id] = 1.5 if i >= rows // 2 else 1.0
                        else:
                            scores[pos_id] = 1.5 if i < rows // 2 else 1.0

        return scores


class SampleMatchModule:
    """
    如果部分揭示與樣本吻合度高，直接用該樣本位置預測 target；否則回空字典。
    """
    def __init__(self):
        # 範例：隨機生成 8×10 樣本，實際請替換為你的 Excel/ZIP 樣本
        N = 8 * 10
        nums = list(range(1, N + 1))
        random.shuffle(nums)
        sample1 = [nums[i*10:(i+1)*10] for i in range(8)]
        self.samples = [sample1]

    def predict(self, grid: list[list[int]], target: int) -> dict[int, float]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            return {}

        best_sample = None
        best_score = -1.0
        for sample in self.samples:
            if len(sample) != rows or len(sample[0]) != cols:
                continue
            match = 0
            total_revealed = 0
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] != -1:
                        total_revealed += 1
                        if sample[i][j] == grid[i][j]:
                            match += 1
            if total_revealed == 0:
                continue
            sc = match / total_revealed
            if sc > best_score:
                best_score = sc
                best_sample = sample
            if sc == 1.0:
                break

        scores: dict[int, float] = {}
        if best_sample and best_score > 0:
            for i in range(rows):
                for j in range(cols):
                    if best_sample[i][j] == target and grid[i][j] == -1:
                        pos_id = i * cols + j + 1
                        scores[pos_id] = 1.0 if best_score == 1.0 else 0.5

        return scores