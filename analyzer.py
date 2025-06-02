# analyzer.py

import numpy as np
from typing import Dict, List

# ── 1. 导入 brain1.py / brain2.py / brain3.py 中定义的所有 exXX 函数 ──
#    下列函数名必须与你在这三个文件里实际定义好的函数一致
from brain1 import (
    ex1, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex16, ex17, ex18, ex19
)
from brain2 import (
    ex10, ex11, ex12, ex13, ex14, ex15
)
from brain3 import (
    ex19 as ex19_b3,  # 如果 brain3.py 也有 ex19，与 brain1 冲突时用别名
    ex20,
    ex21,
    ex22,
    ex23,
    ex24,
    ex25,
    ex26
)

# ── 2. 导入 new_module.py 中新增的 5 个预测模块类 ──
from new_module import (
    ProbabilityModule,
    AdjacencyModule,
    FrequencyModule,
    PatternModule,
    SampleMatchModule,
)

class Analyzer:
    """
    将“原有 exXX 模块”与“新增的 5 个预测模块”整合，
    返回隐藏格的综合评分 {位置ID: 分数}。
    """

    def __init__(self):
        # 1. 保留原本的 exXX 函数
        self.registered_modules = [
            ex1, ex3, ex4, ex5, ex6, ex7, ex8, ex9,
            ex10, ex11, ex12, ex13, ex14, ex15,
            ex16, ex17, ex18, ex19, ex19_b3, ex20,
            ex21, ex22, ex23, ex24, ex25, ex26
        ]

        # 2. 将 5 个新模块实例化
        self.new_modules = [
            ProbabilityModule(),
            AdjacencyModule(),
            FrequencyModule(),
            PatternModule(),
            SampleMatchModule(),
        ]

    def analyze(self, grid: np.ndarray, target: int) -> Dict[int, float]:
        """
        分析整张卡片：
        - grid: 2D numpy array，隐藏格用 -1 表示
        - target: 要预测的数字

        返回 {位置ID: 分数}，位置ID 以 1 开始，逐行优先编号。
        """
        if grid.ndim != 2:
            raise ValueError("analyze: 输入必须是 2 维数组")

        rows, cols = grid.shape

        # 先收集所有隐藏格的位置 ID（1-based，逐行优先）
        hidden_positions: List[int] = [
            i * cols + j + 1
            for i in range(rows)
            for j in range(cols)
            if grid[i, j] == -1
        ]
        if not hidden_positions:
            return {}

        # ── 3. 调用“原有 exXX 模块”，它们返回 numpy.ndarray，
        #       需把对应位置取出来并加到 combined_scores ──
        combined_scores: Dict[int, float] = {}
        for func in self.registered_modules:
            try:
                output_arr = func(grid)  # 接口：func(grid: np.ndarray) -> np.ndarray
            except Exception:
                continue
            if not isinstance(output_arr, np.ndarray):
                continue
            # output_arr.shape 应该与 grid.shape 相同，值为 0 或 1
            for i in range(rows):
                for j in range(cols):
                    if output_arr[i, j] != 0 and grid[i, j] == -1:
                        pos_id = i * cols + j + 1
                        combined_scores[pos_id] = combined_scores.get(pos_id, 0.0) + float(output_arr[i, j])

        # ── 4. 调用“5 个新模块”，它们返回 {posID: raw_score} ──
        temp_scores: Dict[int, float] = {}
        active_new = 0

        for module in self.new_modules:
            try:
                ms: Dict[int, float] = module.predict(grid.tolist(), target)
            except Exception:
                continue
            if not isinstance(ms, dict) or not ms:
                continue

            # 只保留真正隐藏格的位置
            ms = {p: v for p, v in ms.items() if p in hidden_positions}
            total = sum(ms.values())

            if total == 0:
                # 若全为 0，则退回均匀分布
                uniform = 1.0 / len(hidden_positions)
                for p in hidden_positions:
                    ms[p] = uniform
            else:
                # 正规化：使得该模块所有评分之和 = 1
                for p in ms:
                    ms[p] = ms[p] / total

            for p, v in ms.items():
                temp_scores[p] = temp_scores.get(p, 0.0) + float(v)
            active_new += 1

        # 所有新模块都正规化并加到 temp_scores 后，取平均并累加到 combined_scores
        if active_new > 0:
            for p, tot in temp_scores.items():
                avg_score = tot / active_new
                combined_scores[p] = combined_scores.get(p, 0.0) + avg_score

        return combined_scores