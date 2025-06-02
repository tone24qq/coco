# analyzer.py

import numpy as np
from typing import Dict, List

# 1. 导入你在 brain1.py / brain2.py / brain3.py 里实际写的函数
#    这些行必须和你的文件里一模一样，否则会 ImportError
from brain1 import EXT_A2, EXT_M3, EXT_M10
from brain2 import EXT_D3, EXT_F10, EXT_M6
from brain3 import EXT_P4, EXT_R7, EXT_L1

# 2. 导入我写在 new_module.py 中的 5 个模块
from new_module import (
    ProbabilityModule,
    AdjacencyModule,
    FrequencyModule,
    PatternModule,
    SampleMatchModule,
)

class Analyzer:
    """
    整合：先调用 brain1.py/brain2.py/brain3.py 里的 EXT_… 函数，
    得到 {posID: score}；再调用 new_module.py 的 5 个模块同样得到 {posID:score}，
    最后归一化并平均后加到 combined_scores，输出 {posID:综合得分}。
    """

    def __init__(self):
        # 1. 保留你原有的所有 EXT_* 函数
        self.registered_modules = [
            EXT_A2,
            EXT_M3,
            EXT_M10,
            EXT_D3,
            EXT_F10,
            EXT_M6,
            EXT_P4,
            EXT_R7,
            EXT_L1,
        ]
        # 2. 把 5 个新模块实例化
        self.new_modules = [
            ProbabilityModule(),
            AdjacencyModule(),
            FrequencyModule(),
            PatternModule(),
            SampleMatchModule(),
        ]

    def analyze(self, grid: np.ndarray, target: int) -> Dict[int, float]:
        """
        grid: 2D numpy array，-1 表示隐藏格
        target: 要查找的数字
        返回 {posID: 综合分数}
        """
        if grid.ndim != 2:
            raise ValueError("analyze: 输入必须是 2D numpy array")

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

        # ─── 调用“你原来的 EXT_* 函数” ───
        combined_scores: Dict[int, float] = {}
        for func in self.registered_modules:
            try:
                mod_scores: Dict[int, float] = func(grid)  # 假设它们都返回 {posID:score}
            except Exception:
                continue
            if not isinstance(mod_scores, dict):
                continue
            for pos, val in mod_scores.items():
                if pos in hidden_positions:
                    combined_scores[pos] = combined_scores.get(pos, 0.0) + float(val)

        # ─── 调用“我写的 5 个新模块”，再归一化+平均 ───
        temp_scores: Dict[int, float] = {}
        active_new = 0

        for module in self.new_modules:
            try:
                ms: Dict[int, float] = module.predict(grid.tolist(), target)
            except Exception:
                continue
            if not isinstance(ms, dict) or not ms:
                continue

            # 只保留 hidden_positions 中的键
            ms = {p: v for p, v in ms.items() if p in hidden_positions}
            total = sum(ms.values())

            if total == 0:
                # 若所有值为0，就退回均匀分布
                uniform = 1.0 / len(hidden_positions)
                for p in hidden_positions:
                    ms[p] = uniform
            else:
                # 归一化：让该模块输出加起来等于1
                for p in ms:
                    ms[p] = ms[p] / total

            for p, v in ms.items():
                temp_scores[p] = temp_scores.get(p, 0.0) + float(v)
            active_new += 1

        # 把所有新模块平均后加到 combined_scores
        if active_new > 0:
            for p, tot in temp_scores.items():
                avg_score = tot / active_new
                combined_scores[p] = combined_scores.get(p, 0.0) + avg_score

        return combined_scores