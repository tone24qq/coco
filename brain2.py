# brain2.py
"""
brain2.py：放置其他部分 EXT_*_Vec 向量化模組，由新大腦.pdf 搬運。
包含以下函式：
- EXT_GM3_Adv_Connected_Comp_Vec
- EXT_GM4_Spatial_Auto_Corr_Vec
- EXT_GM5_Line_Completion_Vec
- EXT_GM6_Symmetry_Potential_Vec
- EXT_GM7_Numeric_Gaps_Vec
- EXT_GM8_Edge_Affinity_Vec
- EXT_GM9_Center_Control_Vec
- EXT_GM10_Blocking_Value_Vec
- EXT_GM11_Pair_Correlation_Vec
- EXT_GM12_Island_Analysis_Vec
"""

import numpy as np
import math
from collections import Counter, deque
import logging
from typing import List, Any, Optional, Tuple

from brain1 import MathUtils, BoardAnalyzerUtils, logger

# === 9. EXT_GM3_Adv_Connected_Comp_Vec (高級連通元件分析-空格區域) ===

def EXT_GM3_Adv_Connected_Comp_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM3"
    logger.debug("Executing EXT_GM3_Adv_Connected_Comp_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    visited = np.full((rows, cols), False, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if visited[r, c] or grid[r, c] != -1:
                visited[r, c] = True
                continue
            queue = deque([(r, c)])
            region_cells = []
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                region_cells.append((cr, cc))
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == -1:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            size = float(len(region_cells))
            min_r = min(rc[0] for rc in region_cells)
            max_r = max(rc[0] for rc in region_cells)
            min_c = min(rc[1] for rc in region_cells)
            max_c = max(rc[1] for rc in region_cells)
            area = float((max_r - min_r + 1) * (max_c - min_c + 1))
            compactness = size / area if area > 0 else 0.0
            norm_size = MathUtils.normalize_value(size, 1, max(rows, cols), clamp=True)
            norm_comp = MathUtils.normalize_value(compactness, 0, 1.0, clamp=True)
            island_score = (0.5 * norm_size + 0.3 * norm_comp)
            final = MathUtils.normalize_value(island_score, 0, 1.0, clamp=True)
            for (rr, cc) in region_cells:
                scores[rr, cc] = final

    return scores

# === 10. EXT_GM4_Spatial_Auto_Corr_Vec (空間自相關性分析) ===

def EXT_GM4_Spatial_Auto_Corr_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM4"
    logger.debug("Executing EXT_GM4_Spatial_Auto_Corr_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    max_val_on_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_board == 0:
        max_val_on_board = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            neighbor_vals = BoardAnalyzerUtils.get_neighborhood_values(grid, r, c, radius=1, eight_connectivity=True)
            if not neighbor_vals:
                scores[r, c] = 0.0
                continue
            avg = sum(neighbor_vals) / len(neighbor_vals)
            corr = 1.0 - abs(max_val_on_board - avg) / max_val_on_board
            scores[r, c] = MathUtils.normalize_value(corr, 0, 1.0, clamp=True)

    return scores

# === 11. EXT_GM5_Line_Completion_Vec (線段補全) ===

def EXT_GM5_Line_Completion_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM5"
    logger.debug("Executing EXT_GM5_Line_Completion_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0 or min(rows, cols) < 1:
        return scores

    legal_values = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal_values:
        return scores

    score_map = {
        "identical_3": 0.6,
        "arithmetic_3_mend": 0.7,
        "arithmetic_3_extend": 0.5,
    }

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_score = 0.0
            for p_val in legal_values:
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r - dr, c - dc
                    if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        if v1 != -1 and v2 != -1:
                            if p_val == v1 == v2:
                                best_score = max(best_score, score_map["identical_3"])
                            if (v1 + v2) % 2 == 0:
                                mid = (v1 + v2) // 2
                                if mid == p_val and abs(v1 - v2) != 0:
                                    best_score = max(best_score, score_map["arithmetic_3_mend"])
                            if (v1 - p_val) == (p_val - v2) and abs(v1 - p_val) != 0:
                                best_score = max(best_score, score_map["arithmetic_3_extend"])
            scores[r, c] = MathUtils.normalize_value(best_score, 0, 1.0, clamp=True)

    return scores

# === 12. EXT_GM6_Symmetry_Potential_Vec (對稱性潛力) ===

def EXT_GM6_Symmetry_Potential_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM6"
    logger.debug("Executing EXT_GM6_Symmetry_Potential_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    max_val_on_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_board == 0:
        max_val_on_board = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            s_h = 0.0
            mirror_c = cols - 1 - c
            if 0 <= mirror_c < cols and grid[r, mirror_c] != -1:
                s_h = 1.0
            s_v = 0.0
            mirror_r = rows - 1 - r
            if 0 <= mirror_r < rows and grid[mirror_r, c] != -1:
                s_v = 1.0
            s_d1 = 0.0
            if rows == cols and grid[c, r] != -1:
                s_d1 = 1.0
            s_d2 = 0.0
            if rows == cols:
                mr, mc = cols - 1 - c, rows - 1 - r
                if grid[mr, mc] != -1:
                    s_d2 = 1.0
            raw_score = s_h + s_v + s_d1 + s_d2
            scores[r, c] = MathUtils.normalize_value(raw_score, 0, 4.0, clamp=True)

    return scores

# === 13. EXT_GM7_Numeric_Gaps_Vec (數字間隙模式) ===

def EXT_GM7_Numeric_Gaps_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM7"
    logger.debug("Executing EXT_GM7_Numeric_Gaps_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal_values = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal_values:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_gap = 0.0
            for p_val in legal_values:
                left_vals = [grid[r, cc] for cc in range(0, c) if grid[r, cc] != -1]
                right_vals = [grid[r, cc] for cc in range(c+1, cols) if grid[r, cc] != -1]
                if left_vals and right_vals:
                    v1 = left_vals[-1]
                    v2 = right_vals[0]
                    if (v2 - v1) % 2 == 0:
                        mid = (v1 + v2) // 2
                        if mid == p_val and abs(v2 - v1) > 0:
                            best_gap = max(best_gap, 1.0)
                up_vals = [grid[rr, c] for rr in range(0, r) if grid[rr, c] != -1]
                down_vals = [grid[rr, c] for rr in range(r+1, rows) if grid[rr, c] != -1]
                if up_vals and down_vals:
                    v1 = up_vals[-1]
                    v2 = down_vals[0]
                    if (v2 - v1) % 2 == 0:
                        mid = (v1 + v2) // 2
                        if mid == p_val and abs(v2 - v1) > 0:
                            best_gap = max(best_gap, 1.0)
            scores[r, c] = best_gap

    return scores

# === 14. EXT_GM8_Edge_Affinity_Vec (邊緣親和度) ===

def EXT_GM8_Edge_Affinity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM8"
    logger.debug("Executing EXT_GM8_Edge_Affinity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    mode = "prefer_edge"
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            dist_edge = min(r, rows - 1 - r, c, cols - 1 - c)
            max_dist = math.floor((min(rows, cols) - 1) / 2)
            if max_dist <= 0:
                scores[r, c] = 0.0
            else:
                if mode == "prefer_edge":
                    scores[r, c] = MathUtils.normalize_value(max_dist - dist_edge, 0, max_dist, clamp=True)
                else:
                    scores[r, c] = MathUtils.normalize_value(dist_edge, 0, max_dist, clamp=True)

    return scores

# === 15. EXT_GM9_Center_Control_Vec (中心控制偏好) ===

def EXT_GM9_Center_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM9"
    logger.debug("Executing EXT_GM9_Center_Control_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_dist = math.sqrt(center_r**2 + center_c**2)

    mode = "prefer_center"
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            dist = math.sqrt((r - center_r)**2 + (c - center_c)**2)
            if mode == "prefer_center":
                scores[r, c] = MathUtils.normalize_value(max_dist - dist, 0, max_dist, clamp=True)
            else:
                scores[r, c] = MathUtils.normalize_value(dist, 0, max_dist, clamp=True)

    return scores

# === 16. EXT_GM10_Blocking_Value_Vec (阻斷價值評估) ===

def EXT_GM10_Blocking_Value_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM10"
    logger.debug("Executing EXT_GM10_Blocking_Value_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    undesirable_pairs = {(1,1), (1,3)}
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            worst_val = 0.0
            legal_values = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
            for v in legal_values:
                block_score = 0.0
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        if (grid[nr, nc], v) in undesirable_pairs:
                            block_score = max(block_score, 1.0)
                worst_val = max(worst_val, block_score)
            scores[r, c] = 1.0 - MathUtils.normalize_value(worst_val, 0, 1.0, clamp=True)

    return scores

# === 17. EXT_GM11_Pair_Correlation_Vec (數值對相關) ===

def EXT_GM11_Pair_Correlation_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM11"
    logger.debug("Executing EXT_GM11_Pair_Correlation_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_pair = 0.0
            legal_values = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
            for v in legal_values:
                pair_score = 0.0
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        if abs(grid[nr, nc] - v) == 1:
                            pair_score = max(pair_score, 1.0)
                best_pair = max(best_pair, pair_score)
            scores[r, c] = best_pair

    return scores

# === 18. EXT_GM12_Island_Analysis_Vec (島嶼分析) ===

def EXT_GM12_Island_Analysis_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM12"
    logger.debug("Executing EXT_GM12_Island_Analysis_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    visited = np.full((rows, cols), False, dtype=bool)
    max_val_on_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_board == 0:
        max_val_on_board = 1.0

    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            if grid[r, c] == -1:
                visited[r, c] = True
                scores[r, c] = 0.0
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            island_cells = []
            island_vals = []
            while queue:
                cr, cc = queue.popleft()
                island_cells.append((cr, cc))
                island_vals.append(grid[cr, cc])
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] != -1:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            size = float(len(island_cells))
            avg_val = float(sum(island_vals)) / size if size > 0 else 0.0
            min_r = min(cell[0] for cell in island_cells)
            max_r = max(cell[0] for cell in island_cells)
            min_c = min(cell[1] for cell in island_cells)
            max_c = max(cell[1] for cell in island_cells)
            area = float((max_r - min_r + 1) * (max_c - min_c + 1))
            compact = size / area if area > 0 else 0.0

            norm_size = MathUtils.normalize_value(size, 1, rows * cols, clamp=True)
            norm_compact = MathUtils.normalize_value(compact, 0, 1.0, clamp=True)
            norm_avg = MathUtils.normalize_value(avg_val, 1, max_val_on_board, clamp=True)

            island_score = 0.5 * norm_size + 0.3 * norm_compact + 0.2 * norm_avg
            final = MathUtils.normalize_value(island_score, 0, 1.0, clamp=True)

            for (rr, cc) in island_cells:
                scores[rr, cc] = final

    return scores