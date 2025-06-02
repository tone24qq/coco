# analyzer.py

import numpy as np
from typing import Dict, List

# 1. 导入你在 brain1.py 中的所有 ex1–ex9 函数
#    （请务必保证函数名和你 own 文件中一模一样，否则会 ImportError）
from brain1 import ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9

# 2. 导入你在 brain2.py 中的所有 ex10–ex18 函数
from brain2 import ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18

# 3. 导入你在 brain3.py 中的所有 ex19–ex26 函数
from brain3 import ex19, ex20, ex21, ex22, ex23, ex24, ex25, ex26

# 4. 导入 new_module.py 里的 5 个新模块类（新增逻辑）
from new_module import (
    ProbabilityModule,
    AdjacencyModule,
    FrequencyModule,
    PatternModule,
    SampleMatchModule,
)

class Analyzer:
    """
    将“原有 ex1–ex26 函数”与“5 个新模块”整合：
      1. 先调用 ex1–ex26（它们返回与 grid 同形的 np.ndarray）
         把所有在隐藏位（grid==-1）上非零的点计为 1 分累加进 combined_scores；
      2. 再调用 5 个新模块（它们返回 {posID: raw_score}），
         每个先做归一化（和为 1）→ 多个模块累加 → 除以模块数（平均）→ 累加到 combined_scores；
      返回最终 {posID: 综合分数}。
    """

    def __init__(self):
        # 1. 保留旧有的 26 个 ex 函数
        self.registered_modules = [
            ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9,
            ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18,
            ex19, ex20, ex21, ex22, ex23, ex24, ex25, ex26,
        ]
        # 2. 实例化新增的 5 个模块
        self.new_modules = [
            ProbabilityModule(),
            AdjacencyModule(),
            FrequencyModule(),
            PatternModule(),
            SampleMatchModule(),
        ]

    def analyze(self, grid: np.ndarray, target: int) -> Dict[int, float]:
        """
        grid:  2D numpy array，隐藏格用 -1 表示
        target: 要预测的数字

        返回字典 {posID: 综合分数}，posID 从 1 开始，逐行优先编号。
        """
        if grid.ndim != 2:
            raise ValueError("analyze: 输入必须是二维 numpy array")

        rows, cols = grid.shape

        # 先收集所有隐藏格的 posID（1-based，逐行优先）
        hidden_positions: List[int] = [
            i * cols + j + 1
            for i in range(rows)
            for j in range(cols)
            if grid[i, j] == -1
        ]
        if not hidden_positions:
            return {}

        # ─── 第一部分：调用 ex1–ex26，将它们各自返回的 np.ndarray 累加到 combined_scores ───
        combined_scores: Dict[int, float] = {}

        for func in self.registered_modules:
            try:
                # 每个 exX 返回一个与 grid 形状相同的 np.ndarray，值为 0 或 1 或浮点
                output_arr = func(grid)
            except Exception:
                # 如果某个旧函数内部出错，跳过它
                continue

            if not isinstance(output_arr, np.ndarray) or output_arr.shape != grid.shape:
                # 强制要求它们返回的是同形 np.ndarray，否则跳过
                continue

            # 对应在“隐藏位（grid==-1）”上，非零的点认为是 1 分，然后累加
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] == -1 and output_arr[i, j] != 0:
                        pos_id = i * cols + j + 1
                        combined_scores[pos_id] = combined_scores.get(pos_id, 0.0) + 1.0

        # ─── 第二部分：调用 5 个新模块，先做归一化→累加→平均→累加到 combined_scores ───
        temp_scores: Dict[int, float] = {}
        active_new = 0

        for module in self.new_modules:
            try:
                ms: Dict[int, float] = module.predict(grid.tolist(), target)
            except Exception:
                # 如果某个新模块内部出错，跳过它
                continue

            if not isinstance(ms, dict) or not ms:
                continue

            # 只保留真正隐藏格的位置
            ms = {p: v for p, v in ms.items() if p in hidden_positions}
            total = sum(ms.values())

            if total == 0:
                # 如果一个模块输出的所有 raw_score 都是 0，则退回均匀分布
                uniform = 1.0 / len(hidden_positions)
                for p in hidden_positions:
                    ms[p] = uniform
            else:
                # 正规化：该模块所有 ms[p] 相加 = 1
                for p in ms:
                    ms[p] = ms[p] / total

            # 把这个模块的“正规化后分数”累加进 temp_scores
            for p, v in ms.items():
                temp_scores[p] = temp_scores.get(p, 0.0) + float(v)

            active_new += 1

        # 如果至少有一个新模块生效，就把它们的平均分数累加到 combined_scores
        if active_new > 0:
            for p, tot in temp_scores.items():
                avg_score = tot / active_new
                combined_scores[p] = combined_scores.get(p, 0.0) + avg_score

        return combined_scores