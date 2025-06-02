# new_module.py

import math
import random

class ProbabilityModule:
    """
    对所有隐藏格均匀分配概率（基线）。
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
    根据邻近已知数值与 target 的差距打分。
    差距越小，得分越高；如果所有相邻都没贡献，则退回均匀分布（设为 1）。
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
    将所有数字按“低 ≤ N/2”、“高 > N/2”分两类，
    统计四个象限内与 target 同类已揭示数字的数量，
    选择数量最少的象限，将该象限的隐藏格设为权重 2.0，其他象限设为 1.0。
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
    深度检测“行范围”或“列余数”两种排列：
      1. 行范围模式：假设每行是连续数值区间（如 1–10、11–20…）
      2. 列余数模式：假设每列数字末位相同（第 j 列对应末位 j+1，最后一列对应 0）
    如果都不成立，则退回“半区启发式”：若 target > N/2，则偏好下半行，否则偏好上半行。
    """
    def predict(self, grid: list[list[int]], target: int) -> dict[int, float]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            return {}

        range_size = math.ceil((rows * cols) / rows)

        # 检测“行范围模式”
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

        # 检测“列余数模式”
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
    如果部分揭示与已知样本高度匹配，就用该样本位置预测 target；
    否则返回空字典，让其他模块决定。
    """
    def __init__(self):
        # 示例：随机生成一个 8×10 样本；你可以改为加载你的真实样本
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