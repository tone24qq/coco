# analyzer.py

import numpy as np
import inspect
import brain1
import brain2
import brain3
from typing import Dict, List

# ───────────────────────────────────────────────────────────────────────────────────────
# 动态收集：从 brain1.py / brain2.py / brain3.py 中，所有以 "ex" 开头的函数，都纳入旧模块列表。
# 这样无论你究竟定义了 ex1…ex28 中的哪些，都会被自动抓取并执行。
# ───────────────────────────────────────────────────────────────────────────────────────
def _collect_ex_functions(module):
    return [
        obj
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("ex")
    ]

_old_modules = (
    _collect_ex_functions(brain1)
    + _collect_ex_functions(brain2)
    + _collect_ex_functions(brain3)
)

# ────────────────────────────────────────────────────────────────────
# 导入 new_module.py 里的五个“新增模块”类
# ────────────────────────────────────────────────────────────────────
from new_module import (
    ProbabilityModule,
    AdjacencyModule,
    FrequencyModule,
    PatternModule,
    SampleMatchModule,
)


class Analyzer:
    """
    构造时会：
      1. 动态收集 brain1/brain2/brain3 中所有以 "ex" 开头的函数（旧模块）；
      2. 实例化五个新增模块（new_module.py 中的类）；
    分析时会：
      - 先依次执行旧模块 exX(grid)，它们返回与 grid 同形的 np.ndarray；
        在每个隐藏格(grid == -1) 且输出非零时，将 posID 累加 +1 分。
      - 再执行五个新增模块 module.predict(grid_list, target)，它们返回 {posID: raw_score}；
        对每个模块做「模块内部归一化 → 多模块累加 → 平均」后，将平均分叠加到 combined_scores。
      最终返回 {posID: 综合分数}，posID 从 1 开始，逐行优先编号。
    """

    def __init__(self):
        # 1. 保留旧有的 ex 函数列表（动态收集）
        self.registered_modules = _old_modules.copy()
        # 2. 实例化新增的五个模块
        self.new_modules = [
            ProbabilityModule(),
            AdjacencyModule(),
            FrequencyModule(),
            PatternModule(),
            SampleMatchModule(),
        ]

    def analyze(self, grid: np.ndarray, target: int) -> Dict[int, float]:
        """
        grid:  2D numpy array，-1 表示隐藏格
        target: 要预测的数字

        返回 {posID: 综合分数}，posID 从 1 开始，逐行优先编号。
        """
        if grid.ndim != 2:
            raise ValueError("analyze: 输入必须是 2D numpy array")

        rows, cols = grid.shape

        # 收集所有隐藏格的 posID（1-based，逐行优先）
        hidden_positions: List[int] = [
            i * cols + j + 1
            for i in range(rows)
            for j in range(cols)
            if grid[i, j] == -1
        ]
        if not hidden_positions:
            return {}

        # ─── 第一部分：调用旧模块 exX，将每个隐藏格中非零的都 +1 ───
        combined_scores: Dict[int, float] = {}

        for func in self.registered_modules:
            try:
                output_arr = func(grid)
            except Exception:
                # 如果某个旧函数内部抛错，跳过该函数
                continue

            if not isinstance(output_arr, np.ndarray) or output_arr.shape != grid.shape:
                # 强制要求返回与 grid 同形的 ndarray，否则跳过
                continue

            # 在隐藏格(grid==-1) 且 output_arr 非零时，posID 累加 +1
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] == -1 and output_arr[i, j] != 0:
                        pos_id = i * cols + j + 1
                        combined_scores[pos_id] = combined_scores.get(pos_id, 0.0) + 1.0

        # ─── 第二部分：调用五个新增模块 → 「归一化 → 累加 → 平均」───
        temp_scores: Dict[int, float] = {}
        active_new = 0

        for module in self.new_modules:
            try:
                ms: Dict[int, float] = module.predict(grid.tolist(), target)
            except Exception:
                # 如果某个新增模块内部抛错，跳过它
                continue

            if not isinstance(ms, dict) or not ms:
                continue

            # 只保留真正属于 hidden_positions 的键
            ms = {p: v for p, v in ms.items() if p in hidden_positions}
            total = sum(ms.values())

            if total == 0:
                # 如果该模块返回的所有 raw_score 都是 0，就退回均匀分布
                uniform = 1.0 / len(hidden_positions)
                for p in hidden_positions:
                    ms[p] = uniform
            else:
                # 归一化：让该模块内所有 ms[p] 相加 = 1
                for p in ms:
                    ms[p] = ms[p] / total

            # 把这个模块归一化后的分数累加到 temp_scores
            for p, v in ms.items():
                temp_scores[p] = temp_scores.get(p, 0.0) + float(v)

            active_new += 1

        # 若至少一个新增模块生效，就将它们的平均分加到 combined_scores
        if active_new > 0:
            for p, tot in temp_scores.items():
                avg_score = tot / active_new
                combined_scores[p] = combined_scores.get(p, 0.0) + avg_score

        return combined_scores