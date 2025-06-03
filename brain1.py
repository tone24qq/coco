# brain1.py
"""
brain1.py：放置部分 EXT_*_Vec 向量化模組，由新大腦.pdf 搬運。
包含以下函式：
- EXT_A2_Weighted_Proximity_Vec
- EXT_M3_Local_Heterogeneity_Vec
- EXT_D3_Potential_Field_Vec
- EXT_F10_Discontinuity_Vec
- EXT_P7_Pathfinding_Value_Vec
- EXT_R5_Resource_Control_Vec
- EXT_GM1_Row_Control_Vec
- EXT_GM2_Col_Flow_Vec
"""

import numpy as np
import math
from collections import Counter, deque
import logging
from typing import List, Any, Optional, Tuple

# Logging
logger = logging.getLogger(__name__)

# === 工具類別 (摘自新大腦.pdf) ===

class MathUtils:
    @staticmethod
    def sigmoid(x: float, k: float = 1.0) -> float:
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
        if math.isclose(max_val, min_val):
            if math.isclose(value, min_val):
                return 0.5
            elif value < min_val:
                return 0.0
            else:
                return 1.0
        normalized = (value - min_val) / (max_val - min_val)
        if clamp:
            return max(0.0, min(1.0, normalized))
        return normalized

    @staticmethod
    def manhattan_distance(p1: Tuple[int,int], p2: Tuple[int,int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def get_entropy(values: List[Any]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        total_count = len(values)
        entropy = 0.0
        for count in counts.values():
            prob = count / total_count
            entropy -= prob * math.log2(prob)
        return entropy

class BoardAnalyzerUtils:
    @staticmethod
    def get_legal_values_for_placement(grid: np.ndarray) -> set:
        if grid.size == 0:
            return set()
        rows, cols = grid.shape
        all_possible = set(range(1, rows * cols + 1))
        used = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        return all_possible - used

    @staticmethod
    def get_card_max_value_from_grid_dimensions(dim: Tuple[int,int]) -> int:
        return dim[0] * dim[1]

    @staticmethod
    def get_neighborhood_values(
        grid: np.ndarray, r: int, c: int, radius: int = 1,
        eight_connectivity: bool = True, val_func=None, include_center: bool = False
    ) -> List[Any]:
        rows, cols = grid.shape
        values = []
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                if dr == 0 and dc == 0 and not include_center:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = grid[nr, nc]
                    if val != -1:
                        if callable(val_func):
                            values.append(val_func(val))
                        else:
                            values.append(val)
        return values

# === 1. EXT_A2_Weighted_Proximity_Vec (加權鄰近性) ===

def EXT_A2_Weighted_Proximity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_A2"
    logger.debug("Executing EXT_A2_Weighted_Proximity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    radius = 2
    value_weight_factor = 0.1
    distance_decay_factor = 1.5

    max_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_grid == 0:
        max_val_on_grid = 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue
            proximity_score = 0.0

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        num_val = grid[nr, nc]
                        dist = MathUtils.manhattan_distance((r_idx, c_idx), (nr, nc))
                        if dist == 0:
                            continue
                        if dist > radius:
                            continue
                        contribution = (num_val * value_weight_factor) / (dist ** distance_decay_factor)
                        proximity_score += contribution

            num_neighbors = ((2 * radius + 1)**2 - 1)
            heuristic_max = num_neighbors * max_val_on_grid * value_weight_factor / (1**distance_decay_factor)
            if heuristic_max <= 0:
                scores[r_idx, c_idx] = 0.0
            else:
                scores[r_idx, c_idx] = MathUtils.normalize_value(proximity_score, 0, heuristic_max, clamp=True)

    return scores

# === 2. EXT_M3_Local_Heterogeneity_Vec (局部異質性) ===

def EXT_M3_Local_Heterogeneity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_M3"
    logger.debug("Executing EXT_M3_Local_Heterogeneity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    max_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_grid == 0:
        max_val_on_grid = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            neighbor_vals = BoardAnalyzerUtils.get_neighborhood_values(grid, r, c, radius=1, eight_connectivity=True, include_center=False)
            entropy = MathUtils.get_entropy(neighbor_vals)
            scores[r, c] = MathUtils.normalize_value(entropy, 0, math.log2(max(len(neighbor_vals),1)), clamp=True)

    return scores

# === 3. EXT_D3_Potential_Field_Vec (位勢場分析) ===

def EXT_D3_Potential_Field_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_D3"
    logger.debug("Executing EXT_D3_Potential_Field_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    max_influence_radius = 2
    decay_exponent = 1.2
    max_val_on_grid = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_on_grid == 0:
        max_val_on_grid = 1.0

    num_neighbors = ((2 * max_influence_radius + 1)**2 - 1)
    heuristic_max_potential = num_neighbors * max_val_on_grid / (1**decay_exponent)
    if heuristic_max_potential == 0:
        heuristic_max_potential = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            potential = 0.0
            for dr in range(-max_influence_radius, max_influence_radius + 1):
                for dc in range(-max_influence_radius, max_influence_radius + 1):
                    nr, nc = r + dr, c + dc
                    if dr == 0 and dc == 0:
                        continue
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        val = grid[nr, nc]
                        dist = MathUtils.manhattan_distance((r, c), (nr, nc))
                        if dist == 0:
                            continue
                        if dist > max_influence_radius:
                            continue
                        potential += val / (dist ** decay_exponent)
            scores[r, c] = MathUtils.normalize_value(potential, 0, heuristic_max_potential, clamp=True)

    return scores

# === 4. EXT_F10_Discontinuity_Vec (不連續性修復/序列完成度) ===

def EXT_F10_Discontinuity_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_F10"
    logger.debug("Executing EXT_F10_Discontinuity_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal_values = BoardAnalyzerUtils.get_legal_values_for_placement(grid)
    if not legal_values:
        return scores

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            max_seq_score = 0.0
            for p_val in legal_values:
                best_local = 0.0
                directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
                for dr, dc in directions:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r + 2*dr, c + 2*dc
                    if 0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows and 0 <= c2 < cols:
                        v1 = grid[r1, c1]; v2 = grid[r2, c2]
                        if v1 != -1 and v2 != -1:
                            if (v1 - p_val) == (p_val - v2) and abs(v1 - p_val) > 0:
                                best_local = max(best_local, 0.7)
                            if ((v2 - v1) % 2 == 0) and (min(v1, v2) < p_val < max(v1, v2)):
                                diff = (v2 - v1) // 2
                                if v1 + diff == p_val and diff != 0:
                                    best_local = max(best_local, 0.4)
                max_seq_score = max(max_seq_score, best_local)
            scores[r, c] = MathUtils.normalize_value(max_seq_score, 0, 1.0, clamp=True)

    return scores

# === 5. EXT_P7_Pathfinding_Value_Vec (路徑尋找價值) ===

def EXT_P7_Pathfinding_Value_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_P7"
    logger.debug("Executing EXT_P7_Pathfinding_Value_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    legal_values = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal_values:
        return scores

    max_depth = 4
    decay = 1.0
    max_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val == 0:
        max_val = 1.0

    heuristic_max_path = ((2*max_depth + 1)**2 * max_val) / (1**decay)
    if heuristic_max_path == 0:
        heuristic_max_path = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_cell = 0.0
            for v in legal_values:
                max_path_val = 0.0
                visited = set()
                queue = deque([((r, c), 0)])
                while queue:
                    (cr, cc), d = queue.popleft()
                    if d >= max_depth:
                        continue
                    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            cell_val = grid[nr, nc]
                            if cell_val != -1:
                                path_val = cell_val / ((d+1)**decay)
                                max_path_val = max(max_path_val, path_val)
                            else:
                                queue.append(((nr, nc), d+1))
                best_cell = max(best_cell, max_path_val)
            scores[r, c] = MathUtils.normalize_value(best_cell, 0, heuristic_max_path, clamp=True)

    return scores

# === 6. EXT_R5_Resource_Control_Vec (資源控制) ===

def EXT_R5_Resource_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_R5"
    logger.debug("Executing EXT_R5_Resource_Control_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    row_filled = np.sum(grid != -1, axis=1)
    col_filled = np.sum(grid != -1, axis=0)
    max_filled = max(rows, cols)

    w_row = 0.4
    w_col = 0.4
    w_val = 0.2

    legal_values = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not legal_values:
        return scores

    avg_legal = np.mean(legal_values) if legal_values else 0.0
    max_val_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_board == 0:
        max_val_board = 1.0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            row_completion = (row_filled[r] + 1) / cols
            col_completion = (col_filled[c] + 1) / rows
            val_capture = avg_legal / max_val_board
            combined = w_row * row_completion + w_col * col_completion + w_val * val_capture
            scores[r, c] = MathUtils.normalize_value(combined, 0, 1.0, clamp=True)

    return scores

# === 7. EXT_GM1_Row_Control_Vec (行控制力) ===

def EXT_GM1_Row_Control_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM1"
    logger.debug("Executing EXT_GM1_Row_Control_Vec", extra={'request_id': effective_request_id})

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    avg_potential = np.mean(potential) if potential else 0.0
    max_val_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
    if max_val_board == 0:
        max_val_board = 1.0

    for r in range(rows):
        row_vals = [v for v in grid[r, :] if v != -1]
        filled_count = len(row_vals)
        sum_vals = sum(row_vals)
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            row_comp = (filled_count + 1) / cols
            seq_score = 0.0
            if len(row_vals) >= 2:
                diffs = np.diff(sorted(row_vals))
                if np.all(diffs == diffs[0]) and diffs[0] != 0:
                    seq_score = 0.5
            sum_score = sum_vals / max_val_board
            w_row = 0.5; w_sum = 0.3; w_seq = 0.2
            combined = w_row * row_comp + w_sum * sum_score + w_seq * seq_score
            scores[r, c] = MathUtils.normalize_value(combined, 0, 1.0, clamp=True)

    return scores

# === 8. EXT_GM2_Col_Flow_Vec (列流動性/列控制力) ===

def EXT_GM2_Col_Flow_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
    effective_request_id = request_id or "N/A_brain_GM2"
    logger.debug("Executing EXT_GM2_Col_Flow_Vec", extra={'request_id': effective_request_id}')

    return EXT_GM1_Row_Control_Vec(grid.T, request_id).T