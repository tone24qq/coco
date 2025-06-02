# analyzer.py

import numpy as np
from typing import Dict, List

# ────────────────────────────────────────────────────────────────────
# 一律「保留並調用」你自己在 brain1.py / brain2.py / brain3.py 中寫的 ex 函數：
#
#    brain1.py  ： ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9
#    brain2.py  ： ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18
#    brain3.py  ： ex19, ex20, ex21, ex22, ex23, ex24, ex25, ex26
#
# 請確保這些名稱和你手上三個檔案裡一模一樣，否則會 import 失敗。
# 這裡**絕對不動**你的原本實作，只是將它們全部匯入並執行一次。
# ────────────────────────────────────────────────────────────────────
from brain1 import ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9
from brain2 import ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18
from brain3 import ex19, ex20, ex21, ex22, ex23, ex24, ex25, ex26

# ────────────────────────────────────────────────────────────────────
# 再把我寫好的五個「新增模組」匯入，放在 new_module.py：
#   ProbabilityModule
#   AdjacencyModule
#   FrequencyModule
#   PatternModule
#   SampleMatchModule
# 這五個模組不會影響到你原本 ex1~ex26 的任何實作。
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
    統一調用：
      1. 你原本在 brain1.py/brain2.py/brain3.py 裡的 ex1~ex26，一次跑完，把每個 隱藏格(grid==-1) 上
         nonzero 的位置都 +1 分，累計到 combined_scores 。
      2. 再調用五個新增模組（new_module.py 內的那五個）。它們各自返回 {posID: raw_score}，
         先做「該模組內部歸一化 → 再五個模組平均 → 最後累加到 combined_scores」。
      回傳最終 {posID: 綜合分數}。
    """

    def __init__(self):
        # 正式「保留」舊有的 26 支 ex 函數
        self.registered_modules = [
            ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9,
            ex10, ex11, ex12, ex13, ex14, ex15, ex16, ex17, ex18,
            ex19, ex20, ex21, ex22, ex23, ex24, ex25, ex26,
        ]
        # 再「初始化」我寫的 5 個新模組
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

        返回 {posID: 綜合分數}（posID 從 1 開始，逐行優先編號）。
        """
        if grid.ndim != 2:
            raise ValueError("analyze: 輸入必須是 2D numpy array")

        rows, cols = grid.shape

        # 先收集所有隐藏格的 posID（1-based，逐行優先）
        hidden_positions: List[int] = [
            i * cols + j + 1
            for i in range(rows)
            for j in range(cols)
            if grid[i, j] == -1
        ]
        if not hidden_positions:
            return {}

        # ─── 第一部分：調用 ex1~ex26 舊函數，累加到 combined_scores ───
        combined_scores: Dict[int, float] = {}

        for func in self.registered_modules:
            try:
                output_arr = func(grid)
            except Exception:
                # 如果某個舊函數執行出錯，跳過它
                continue

            if not isinstance(output_arr, np.ndarray) or output_arr.shape != grid.shape:
                # 強制要求它回傳與 grid 同形的 ndarray，否則略過
                continue

            # 只在隐藏格(grid==-1) 且 output_arr 非零時 +1
            for i in range(rows):
                for j in range(cols):
                    if grid[i, j] == -1 and output_arr[i, j] != 0:
                        pos_id = i * cols + j + 1
                        combined_scores[pos_id] = combined_scores.get(pos_id, 0.0) + 1.0

        # ─── 第二部分：調用 5 個新模組 → 「歸一化 → 累加 → 平均」───
        temp_scores: Dict[int, float] = {}
        active_new = 0

        for module in self.new_modules:
            try:
                ms: Dict[int, float] = module.predict(grid.tolist(), target)
            except Exception:
                continue

            if not isinstance(ms, dict) or not ms:
                continue

            # 只保留真正属于 hidden_positions 的鍵
            ms = {p: v for p, v in ms.items() if p in hidden_positions}
            total = sum(ms.values())

            if total == 0:
                # 若一個模組全給 0，就退回「對所有 hidden 全部給 1/len(hidden)」
                uniform = 1.0 / len(hidden_positions)
                for p in hidden_positions:
                    ms[p] = uniform
            else:
                # 正規化：讓這個模組的 ms[p] 相加 = 1
                for p in ms:
                    ms[p] = ms[p] / total

            # 把「正規化後的分數」先累加到 temp_scores
            for p, v in ms.items():
                temp_scores[p] = temp_scores.get(p, 0.0) + float(v)

            active_new += 1

        # 最後把所有新模組的平均分數累加到 combined_scores
        if active_new > 0:
            for p, tot in temp_scores.items():
                avg_score = tot / active_new
                combined_scores[p] = combined_scores.get(p, 0.0) + avg_score

        return combined_scores