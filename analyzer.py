# analyzer.py

import numpy as np
from typing import Dict, List

# 匯入你原本在 brain1.py、brain2.py、brain3.py 等檔案中定義的函式
# 請確認函式名稱與你實際檔案中的一致
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

def analyze_full_board(grid: np.ndarray, target: int) -> Dict[int, float]:
    """
    結合舊有的 EXT_* 模組與新加入的 5 個預測模組，
    回傳每個隱藏格的綜合分數：{位置ID: 分數}。

    grid: 2D numpy array，-1 表示未開格
    target: 要預測的號碼
    """
    if grid.ndim != 2:
        raise ValueError("analyze_full_board: 輸入必須是 2 維陣列")

    rows, cols = grid.shape

    # 蒐集所有隱藏格的位置 ID（1-based，逐行優先）
    hidden_positions: List[int] = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == -1:
                hidden_positions.append(i * cols + j + 1)
    if not hidden_positions:
        return {}

    # ─── 舊有 EXT_* 模組部分 ───
    combined_scores: Dict[int, float] = {}
    for func in (EXT_A2, EXT_M3, EXT_M10, EXT_D3, EXT_F10, EXT_M6, EXT_P4, EXT_R7, EXT_L1):
        try:
            mod_scores = func(grid)  # 假設介面是 func(grid) -> {pos_id: raw_score}
        except Exception:
            continue
        if not isinstance(mod_scores, dict):
            continue
        for pos, val in mod_scores.items():
            combined_scores[pos] = combined_scores.get(pos, 0.0) + float(val)

    # ─── 新模組部分 ───
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
            ms = module.predict(grid.tolist(), target)
        except Exception:
            continue
        if not isinstance(ms, dict) or not ms:
            continue

        # 只保留在 hidden_positions 中的 key
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