# analyzer.py

import numpy as np
from typing import Dict, List

# 匯入原本你在 brain1.py、brain2.py、brain3.py 等檔案裡定義的函式
# 請確保函式名稱與你實際檔案中的一致
from brain1 import EXT_A2, EXT_M3, EXT_M10
from brain2 import EXT_D3, EXT_F10, EXT_M6
from brain3 import EXT_P4, EXT_R7, EXT_L1

# 匯入 new_module.py 裡新增的五個模組類別
from new_module import (
    ProbabilityModule,
    AdjacencyModule,
    FrequencyModule,
    PatternModule,
    SampleMatchModule,
)

def analyze_full_board(grid: np.ndarray) -> Dict[int, float]:
    """
    統一呼叫「舊有的 EXT_* 模組」與「五個新預測模組」，
    最後回傳合併後的分數字典 {位置ID: 分數}。

    grid: 2D numpy array，隱藏格用 -1 表示。
    """
    if grid.ndim != 2:
        raise ValueError("analyze_full_board: 輸入必須是 2 維陣列")

    rows, cols = grid.shape

    # 先蒐集所有隱藏格的位置 ID（1-based，逐行優先）
    hidden_positions: List[int] = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == -1:
                hidden_positions.append(i * cols + j + 1)

    if not hidden_positions:
        return {}

    # ─────────────── 舊有 EXT_* 模組部分 ───────────────
    combined_scores: Dict[int, float] = {}
    for func in (EXT_A2, EXT_M3, EXT_M10, EXT_D3, EXT_F10, EXT_M6, EXT_P4, EXT_R7, EXT_L1):
        # 假設每個 EXT_* 函式的介面都是： func(grid) -> dict[int, float]
        try:
            mod_scores = func(grid)  # 取得該模組回傳的 {pos_id: raw_score}
        except Exception:
            # 若某個舊模組執行時出錯，就跳過它
            continue
        if not isinstance(mod_scores, dict):
            continue
        for pos, val in mod_scores.items():
            combined_scores[pos] = combined_scores.get(pos, 0.0) + float(val)

    # ─────────────── 新模組部分 ───────────────
    new_modules = [
        ProbabilityModule(),
        AdjacencyModule(),
        FrequencyModule(),
        PatternModule(),
        SampleMatchModule(),
    ]

    temp_scores: Dict[int, float] = {}
    active_new = 0

    for module in new_modules:
        try:
            ms = module.predict(grid.tolist(), None if module.__class__.__name__ == "ProbabilityModule" else None)
            # 這裡以 None 作為 target 傳入 ProbabilityModule，但實際所有新模組都需要 target
            # 所以請在此處把 None 改成實際需要的目標值（如果在 analyze_full_board 中不需要 target，可以改成其他寫法）
        except Exception:
            continue
        # ms 要是 dict[int, float]
        if not isinstance(ms, dict) or not ms:
            continue

        # 只保留確實隱藏格的位置
        ms = {p: v for p, v in ms.items() if p in hidden_positions}
        total = sum(ms.values())

        if total == 0:
            uniform = 1.0 / len(hidden_positions)
            for p in hidden_positions:
                ms[p] = uniform
        else:
            for p in ms:
                ms[p] = ms[p] / total

        for p, v in ms.items():
            temp_scores[p] = temp_scores.get(p, 0.0) + float(v)

        active_new += 1

    if active_new > 0:
        for p, tot in temp_scores.items():
            avg_score = tot / active_new
            combined_scores[p] = combined_scores.get(p, 0.0) + avg_score

    return combined_scores