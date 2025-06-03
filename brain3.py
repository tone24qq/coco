# brain3.py
"""
brain3.py：放置其餘 EXT_*_Vec 向量化模組，由新大腦.pdf 搬運。
包含以下函式：
- EXT_GM13_Sequence_Diversity_Vec
- EXT_GM14_Risk_Assessment_Vec
- EXT_GM15_Information_Gain_Vec
- EXT_GM16_Harmonic_Centrality_Vec
- EXT_GM17_Entropy_Minimization_Vec
- EXT_GM18_RL_Value_Est_Vec
- EXT_GM19_Masked_Number_Skip_Pattern_Vec
- EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec
"""

import numpy as np
import math
from collections import Counter, deque
import logging
from typing import List, Any, Optional, Tuple

from brain1 import MathUtils, BoardAnalyzerUtils, logger

# === 19. EXT_GM13_Sequence_Diversity_Vec (序列多樣性) ===

def EXT_GM13_Sequence_Diversity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM13"
    logger.debug("Executing EXT_GM13_Sequence_Diversity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            diversity = 0.0
            potential = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
            if not potential:
                scores[r, c] = 0.0
                continue
            for v in potential:
                # 檢查上下
                if 0 < r < rows-1 and grid[r-1, c] != -1 and grid[r+1, c] != -1:
                    if (grid[r-1, c] + grid[r+1, c]) % 2 == 0:
                        mid = (grid[r-1, c] + grid[r+1, c]) // 2
                        if mid == v:
                            diversity += 0.5
                # 檢查左右
                if 0 < c < cols-1 and grid[r, c-1] != -1 and grid[r, c+1] != -1:
                    if (grid[r, c-1] + grid[r, c+1]) % 2 == 0:
                        mid = (grid[r, c-1] + grid[r, c+1]) // 2
                        if mid == v:
                            diversity += 0.5
            scores[r, c] = MathUtils.normalize_value(diversity, 0, 1.0, clamp=True)

    return scores

# === 20. EXT_GM14_Risk_Assessment_Vec (風險評估) ===

def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM14"
    logger.debug("Executing EXT_GM14_Risk_Assessment_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    risky_patterns = {(2,4), (3,5)}
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            worst_risk = 0.0
            legal = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
            for v in legal:
                risk = 0.0
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        if (grid[nr, nc], v) in risky_patterns:
                            risk = max(risk, 1.0)
                worst_risk = max(worst_risk, risk)
            scores[r, c] = 1.0 - MathUtils.normalize_value(worst_risk, 0, 1.0, clamp=True)
    return scores

# === 21. EXT_GM15_Information_Gain_Vec (資訊增益評估) ===

def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM15"
    logger.debug("Executing EXT_GM15_Information_Gain_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    flat_vals = [int(v) for v in grid.flatten()]
    initial_entropy = MathUtils.get_entropy(flat_vals)

    legal = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal:
        return scores

    max_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            max_entropy_reduction = 0.0
            for v in legal:
                temp = grid.copy()
                temp[r, c] = v
                flat_temp = [int(x) for x in temp.flatten()]
                ent_after = MathUtils.get_entropy(flat_temp)
                reduction = initial_entropy - ent_after
                if reduction > max_entropy_reduction:
                    max_entropy_reduction = reduction
            max_possible_entropy = math.log2(rows*cols) if rows*cols > 1 else 1.0
            scores[r, c] = MathUtils.normalize_value(max_entropy_reduction, 0, max_possible_entropy, clamp=True)

    return scores

# === 22. EXT_GM16_Harmonic_Centrality_Vec (調和中心性) ===

def EXT_GM16_Harmonic_Centrality_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM16"
    logger.debug("Executing EXT_GM16_Harmonic_Centrality_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    filled_positions = [(i, j) for i in range(rows) for j in range(cols) if grid[i, j] != -1]
    max_score = 0.0
    raw = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                raw[r, c] = 0.0
                continue
            hcentral = 0.0
            for (ri, ci) in filled_positions:
                dist = MathUtils.manhattan_distance((r, c), (ri, ci))
                if dist > 0:
                    hcentral += 1.0 / dist
            raw[r, c] = hcentral
            if hcentral > max_score:
                max_score = hcentral

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -1:
                scores[r, c] = MathUtils.normalize_value(raw[r, c], 0, max_score, clamp=True)
            else:
                scores[r, c] = 0.0

    return scores

# === 23. EXT_GM17_Entropy_Minimization_Vec (熵最小化) ===

def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM17"
    logger.debug("Executing EXT_GM17_Entropy_Minimization_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    flat_vals = [int(v) for v in grid.flatten()]
    initial_global_entropy = MathUtils.get_entropy(flat_vals)
    max_possible_entropy = math.log2(rows*cols) if rows*cols > 1 else 1.0

    legal = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_gain = 0.0
            for v in legal:
                temp = grid.copy()
                temp[r, c] = v
                flat_temp = [int(x) for x in temp.flatten()]
                g_ent = MathUtils.get_entropy(flat_temp)
                local_vals_prev = BoardAnalyzerUtils.get_neighborhood_values(grid, r, c, radius=1, eight_connectivity=True, include_center=True)
                local_ent_prev = MathUtils.get_entropy(local_vals_prev)
                local_vals = BoardAnalyzerUtils.get_neighborhood_values(temp, r, c, radius=1, eight_connectivity=True, include_center=True)
                local_ent = MathUtils.get_entropy(local_vals)
                gain = (initial_global_entropy - g_ent) + (0.5 * (local_ent_prev - local_ent))
                best_gain = max(best_gain, gain)
            scores[r, c] = MathUtils.normalize_value(best_gain, 0, max_possible_entropy + 0.5 * max_possible_entropy, clamp=True)

    return scores

# === 24. EXT_GM18_RL_Value_Est_Vec (類強化學習價值估計) ===

def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM18"
    logger.debug("Executing EXT_GM18_RL_Value_Est_Vec", extra={'request_id': effective_request_id}')

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not potential:
        return scores

    FEATURE_WEIGHTS = {
        "identical_3": 1.0,
        "arithmetic_3": 0.7,
        "center_control": 0.5
    }
    max_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val == 0:
        max_val = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_val = 0.0
            for v in potential:
                feat1 = 0.0
                if 0 < c < cols-1 and grid[r, c-1] == v == grid[r, c+1] and grid[r, c-1] != -1:
                    feat1 = 1.0
                feat2 = 0.0
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                        if grid[r1, c1] != -1 and grid[r2, c2] != -1:
                            if (grid[r1, c1] + grid[r2, c2]) % 2 == 0 and (grid[r1, c1] + grid[r2, c2]) // 2 == v and abs(grid[r1, c1] - grid[r2, c2]) > 0:
                                feat2 = 1.0
                center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
                dist = math.sqrt((r - center_r)**2 + (c - center_c)**2)
                feat3 = 1.0 - (dist / math.sqrt(center_r**2 + center_c**2)) if rows == cols else 0.0

                val_score = (FEATURE_WEIGHTS["identical_3"] * feat1 +
                             FEATURE_WEIGHTS["arithmetic_3"] * feat2 +
                             FEATURE_WEIGHTS["center_control"] * feat3)
                best_val = max(best_val, val_score)
            max_possible = sum(FEATURE_WEIGHTS.values())
            scores[r, c] = MathUtils.normalize_value(best_val, 0, max_possible, clamp=True)

    return scores

# === 25. EXT_GM19_Masked_Number_Skip_Pattern_Vec (遮蔽號跳躍模式) ===

def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM19"
    logger.debug("Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec", extra={'request_id': effective_request_id}')

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_skip = 0.0
            legal = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
            for v in legal:
                skip_score = 0.0
                if 0 < c < cols-1:
                    left = grid[r, c-1]
                    right = grid[r, c+1]
                    if left != -1 and right != -1:
                        if (right - left) == 2 * (v - left):
                            skip_score = 1.0
                best_skip = max(best_skip, skip_score)
            scores[r, c] = best_skip

    return scores

# === 26. EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec (內部間隙補全獎勵) ===

def EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM20"
    logger.debug("Executing EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec", extra={'request_id': effective_request_id}')

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_bonus = 0.0
            for v in legal:
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        if v1 != -1 and v2 != -1:
                            if (v1 + v2) % 2 == 0 and (v1 + v2) // 2 == v and abs(v1 - v2) != 0:
                                best_bonus = max(best_bonus, 0.1)
            scores[r, c] = MathUtils.normalize_value(best_bonus, 0, 0.1, clamp=True)

    return scores