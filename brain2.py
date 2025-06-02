# brain2.py
# Part 2 of 3: Contains the second set of AI scoring modules.
# 來源：Brain.txt, 新大腦.pdf, 给你2025资料在深度建议一次.pdf, 极限强化.pdf

# 來源：知識大典.txt – 防錯字典.txt – "PEP 8 代码风格指南" – "導入順序"
# 1. 標準庫導入
import logging
import math
import uuid # For fallback request_id
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Set, Tuple # PEP 604 via | None

# 2. 第三方庫導入
import numpy as np
from pydantic import BaseModel, Field

# 3. 本地應用/自定义模块导入
# 來源：知識大典.txt – 防錯字典.txt – "ImportError" (防範：確保 brain1 存在且包含必要定義)
try:
    from brain1 import BaseModuleConfig, MathUtils, BoardAnalyzerUtils # Essential utilities from brain1
except ImportError as e:
    logging.critical(f"CRITICAL: Failed to import essential components from brain1.py: {e}. brain2.py cannot function.", exc_info=True)
    raise

# --- Logging Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid duplicate handlers
    logger.addHandler(logging.NullHandler())

# --- Pydantic Config Models for Modules (brain2: GM4-GM12) ---
# 引用：知識大典.txt – 2024-2025知識全集.txt - "3.1.2 Pydantic v2 完整遷移指南" (使用BaseModel, Field)

class SpatialAutocorrelationConfig(BaseModuleConfig): # For GM4
    autocorrelation_type: str = Field(default="positive", pattern="^(positive|negative)$")
    neighborhood_radius: int = Field(default=1, ge=1)
    use_median_for_hypothetical: bool = Field(default=True)

class LineCompletionConfig(BaseModuleConfig): # For GM5
    target_line_length: int = Field(default=3, ge=3)
    score_identical_3: float = Field(default=0.6, ge=0.0)
    score_arithmetic_3_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_3_extend: float = Field(default=0.5, ge=0.0)
    enable_quality_enhancement: bool = Field(default=True)
    score_arithmetic_3_mend_high_val_bonus: float = Field(default=0.2, ge=0.0)
    high_value_threshold_factor_gm5: float = Field(default=0.66, ge=0, le=1)

class SymmetryPotentialConfig(BaseModuleConfig): # For GM6
    score_horizontal: float = Field(default=0.7, ge=0.0)
    score_vertical: float = Field(default=0.7, ge=0.0)
    score_point_center: float = Field(default=0.8, ge=0.0)
    score_main_diagonal: float = Field(default=0.6, ge=0.0)
    score_anti_diagonal: float = Field(default=0.6, ge=0.0)
    strict_square_for_diagonal: bool = Field(default=True)

class NumericGapsConfig(BaseModuleConfig): # For GM7
    score_arithmetic_1_gap_fill: float = Field(default=0.9, ge=0.0)
    score_arithmetic_generic_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_generic_extend: float = Field(default=0.5, ge=0.0)
    enable_quality_enhancement_gm7: bool = Field(default=True)
    score_gap_fill_high_val_bonus: float = Field(default=0.1, ge=0.0)
    high_value_threshold_factor_gm7: float = Field(default=0.66, ge=0, le=1)

class EdgeAffinityConfig(BaseModuleConfig): # For GM8
    affinity_mode: str = Field(default="prefer_edge", pattern="^(prefer_edge|avoid_edge)$")
    corner_bonus_prefer: float = Field(default=0.2, ge=0.0)
    corner_penalty_avoid: float = Field(default=0.2, ge=0.0)

class CenterControlConfig(BaseModuleConfig): # For GM9
    affinity_mode: str = Field(default="prefer_center", pattern="^(prefer_center|avoid_center)$")

class BlockingValueConfig(BaseModuleConfig): # For GM10
    # 來源：Brain2.txt (source 457-458)
    undesirable_sequences_list: List[List[int]] = Field(default_factory=lambda: [[1, 1, 1], [2, 2, 2]])
    score_if_safe: float = Field(default=0.9, ge=0.0, le=1.0)
    score_if_unsafe: float = Field(default=0.1, ge=0.0, le=1.0)
    check_line_length: int = Field(default=3, ge=2)

class PairCorrelationConfig(BaseModuleConfig): # For GM11
    # 來源：Brain2.txt (source 459)
    favorable_pairs: Dict[Tuple[int, int], float] = Field(default_factory=lambda: {
        (3, 7): 0.8, (7, 3): 0.8, (1, 2): 0.6, (2, 1): 0.6, (10,20):0.7, (20,10):0.7
    })

class IslandAnalysisConfig(BaseModuleConfig): # For GM12
    # 來源：Brain2.txt (source 460)
    w_size: float = Field(default=0.4, ge=0.0, le=1.0)
    w_compactness: float = Field(default=0.3, ge=0.0, le=1.0)
    w_avg_value: float = Field(default=0.3, ge=0.0, le=1.0)

# --- Scoring Module Implementations (brain2: GM4-GM12) ---

# 引用：建議.txt (source 651, 706) - 鄰域操作的向量化 (卷積或填充切片思路)
def EXT_GM4_Spatial_Auto_Corr_Vec(
    grid: np.ndarray,
    config: SpatialAutocorrelationConfig,
    request_id: str | None = "N/A_GM4_SpatialAutoCorr", # PEP 604
) -> np.ndarray:
    """(GM4 - 空間自相關性分析) 來源：新大腦.pdf (Page 23)"""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm4-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_GM4 with config: {config.model_dump_json(indent=2)}", extra=log_extra)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores * config.weight # Apply weight on early return

    potential_numbers = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))

    hypothetical_val_to_place: float
    if potential_numbers:
        if config.use_median_for_hypothetical:
            hypothetical_val_to_place = float(np.median(potential_numbers))
        else:
            hypothetical_val_to_place = float(np.mean(potential_numbers))
    else:
        max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
        hypothetical_val_to_place = (1.0 + float(max_board_val)) / 2.0 if max_board_val > 0 else 0.5

    max_val_for_norm = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)))
    if max_val_for_norm == 0: max_val_for_norm = 1.0

    empty_r_indices, empty_c_indices = np.where(grid == -1)

    # Optimized neighborhood value retrieval using padding
    # 引用：知識大典.txt – 2024-2025知識全集.txt – "4.1.1 SIMD 加速與向量化運算優化" (NumPy's pad is efficient)
    padded_grid_for_neighbors = np.pad(grid.astype(float), config.neighborhood_radius, mode='constant', constant_values=np.nan)

    for r_idx, c_idx in zip(empty_r_indices, empty_c_indices):
        pr, pc = r_idx + config.neighborhood_radius, c_idx + config.neighborhood_radius
        window = padded_grid_for_neighbors[
            pr - config.neighborhood_radius : pr + config.neighborhood_radius + 1,
            pc - config.neighborhood_radius : pc + config.neighborhood_radius + 1
        ]
        # Mask to exclude the center cell and NaN (padded) values
        mask = np.ones(window.shape, dtype=bool)
        mask[config.neighborhood_radius, config.neighborhood_radius] = False # Exclude center
        
        valid_neighbor_values = window[mask & ~np.isnan(window) & (window != -1)] # Filter out -1 as well

        if valid_neighbor_values.size == 0:
            scores[r_idx, c_idx] = 0.5  # Neutral score if no valid neighbors
            continue

        mean_neighbors = np.mean(valid_neighbor_values)
        diff_hypo_to_mean = abs(hypothetical_val_to_place - mean_neighbors)
        norm_diff = MathUtils.normalize_value(diff_hypo_to_mean, 0, max_val_for_norm, clamp=True)
        
        current_score = (1.0 - norm_diff) if config.autocorrelation_type == "positive" else norm_diff
        scores[r_idx, c_idx] = current_score
            
    return scores * config.weight

# 引用：建議.txt (source 709) - 序列查找 (find_sequences_in_line) 複雜，保持但優化周邊
def EXT_GM5_Line_Completion_Vec(
    grid: np.ndarray,
    config: LineCompletionConfig,
    request_id: str | None = "N/A_GM5_LineComp",
) -> np.ndarray:
    """(GM5-線段補全) 來源：新大腦.pdf (Page 24)"""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm5-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_GM5 with config: {config.model_dump_json(indent=2)}", extra=log_extra)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    # 來源：Brain2.txt (source 472) - improved boundary check
    if rows == 0 or cols == 0 or config.target_line_length > max(rows,cols) : 
        return scores * config.weight

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores * config.weight

    max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))
    if max_board_val == 0 : max_board_val = 1.0

    empty_r_indices, empty_c_indices = np.where(grid == -1)

    for r_idx, c_idx in zip(empty_r_indices, empty_c_indices):
        max_completion_score_for_cell: float = 0.0
        for p_val in potential_numbers_to_place:
            current_pval_max_score_contribution: float = 0.0
            for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                for i_offset in range(config.target_line_length):
                    current_line_values: List[int] = []
                    is_valid_line_segment = True
                    for k_in_segment in range(config.target_line_length):
                        eval_r, eval_c = r_idx + (k_in_segment - i_offset) * dr_dir, c_idx + (k_in_segment - i_offset) * dc_dir
                        if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                            is_valid_line_segment = False; break
                        current_line_values.append(p_val if (eval_r, eval_c) == (r_idx, c_idx) else int(grid[eval_r, eval_c]))
                    
                    if is_valid_line_segment and all(val != -1 for val in current_line_values):
                        s = current_line_values
                        temp_score = 0.0
                        if len(set(s)) == 1: temp_score = max(temp_score, config.score_identical_3)
                        if len(s) >= 2:
                            # 引用：知識大典.txt – 2024-2025知識全集.txt – "4.1 NumPy 2.0 新功能深度解析" (np.diff for efficiency)
                            diffs = np.diff(s).tolist() 
                            if diffs and len(set(diffs)) == 1 and diffs[0] != 0:
                                if config.target_line_length == 3:
                                    if i_offset == 1: # Mend
                                        temp_score = max(temp_score, config.score_arithmetic_3_mend)
                                        if config.enable_quality_enhancement and (sum(s) / len(s)) > (max_board_val * config.high_value_threshold_factor_gm5):
                                            temp_score += config.score_arithmetic_3_mend_high_val_bonus
                                    else: temp_score = max(temp_score, config.score_arithmetic_3_extend) # Extend
                                else: temp_score = max(temp_score, config.score_arithmetic_3_mend) # Default
                        current_pval_max_score_contribution = max(current_pval_max_score_contribution, temp_score)
            max_completion_score_for_cell = max(max_completion_score_for_cell, current_pval_max_score_contribution)
        scores[r_idx, c_idx] = MathUtils.normalize_value(max_completion_score_for_cell, 0, 1.0, clamp=True) # Max possible score is ~1.0 + bonus
    return scores * config.weight

# 引用：建議.txt (source 654, 709) - np.indices for coordinate-based logic
def EXT_GM6_Symmetry_Potential_Vec(
    grid: np.ndarray,
    config: SymmetryPotentialConfig,
    request_id: str | None = "N/A_GM6_Symmetry",
) -> np.ndarray:
    """(GM6-對稱性潛力) 來源：新大腦.pdf (Page 27)"""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm6-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_GM6 with config: {config.model_dump_json(indent=2)}", extra=log_extra)

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores * config.weight

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))
    if not potential_numbers_to_place: return scores * config.weight

    empty_r_indices, empty_c_indices = np.where(grid == -1)
    # Pre-calculate symmetric coordinates for all empty cells for relevant symmetry types
    # This avoids redundant calculations inside the p_val loop.

    # Horizontal symmetry points for all empty cells
    sym_h_c = cols - 1 - empty_c_indices

    # Vertical symmetry points
    sym_v_r = rows - 1 - empty_r_indices

    # Point center symmetry points
    sym_p_r = rows - 1 - empty_r_indices
    sym_p_c = cols - 1 - empty_c_indices
    
    # Main diagonal symmetry points (r,c) -> (c,r)
    sym_d1_r, sym_d1_c = empty_c_indices.copy(), empty_r_indices.copy() # Need copy for reassignment

    # Anti-diagonal symmetry points for square grids (r,c) -> (N-1-c, N-1-r)
    sym_d2_r, sym_d2_c = (cols - 1 - empty_c_indices), (rows - 1 - empty_r_indices)


    for i, (r_idx, c_idx) in enumerate(zip(empty_r_indices, empty_c_indices)):
        max_symmetry_score_for_cell: float = 0.0
        for p_val in potential_numbers_to_place:
            current_pval_max_sym: float = 0.0
            
            # Horizontal
            shc = sym_h_c[i]
            if shc != c_idx and (0 <= r_idx < rows and 0 <= shc < cols and grid[r_idx, shc] == p_val):
                current_pval_max_sym = max(current_pval_max_sym, config.score_horizontal)
            # Vertical
            svr = sym_v_r[i]
            if svr != r_idx and (0 <= svr < rows and 0 <= c_idx < cols and grid[svr, c_idx] == p_val):
                current_pval_max_sym = max(current_pval_max_sym, config.score_vertical)
            # Point Center
            spr, spc = sym_p_r[i], sym_p_c[i]
            if (spr != r_idx or spc != c_idx) and (0 <= spr < rows and 0 <= spc < cols and grid[spr, spc] == p_val):
                current_pval_max_sym = max(current_pval_max_sym, config.score_point_center)
            
            if not config.strict_square_for_diagonal or rows == cols:
                # Main Diagonal
                sd1r, sd1c = sym_d1_r[i], sym_d1_c[i] # These are c_idx, r_idx
                if (sd1r != r_idx or sd1c != c_idx) and (0 <= sd1r < rows and 0 <= sd1c < cols and grid[sd1r, sd1c] == p_val):
                    current_pval_max_sym = max(current_pval_max_sym, config.score_main_diagonal)
                # Anti-Diagonal (only if square for this simplified vectorized version)
                if rows == cols: # Simplified to square only for this type of anti-diagonal reflection
                    sd2r, sd2c = sym_d2_r[i], sym_d2_c[i] # These are (cols-1-c_idx), (rows-1-r_idx)
                    if (sd2r != r_idx or sd2c != c_idx) and (0 <= sd2r < rows and 0 <= sd2c < cols and grid[sd2r, sd2c] == p_val):
                        current_pval_max_sym = max(current_pval_max_sym, config.score_anti_diagonal)
            
            if current_pval_max_sym > max_symmetry_score_for_cell:
                max_symmetry_score_for_cell = current_pval_max_sym
        scores[r_idx, c_idx] = MathUtils.normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True)
    return scores * config.weight


# 來源：新大腦.pdf - 13. EXT_GM7_Numeric_Gaps_Vec (Page 29-32)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM7強化建議
def EXT_GM7_Numeric_Gaps_Vec(
    grid: np.ndarray,
    config: NumericGapsConfig,
    request_id: str | None = "N/A_GM7_NumGaps",
) -> np.ndarray:
    """
    (GM7 - 數值間隙填充) 評估填補數字間隙的價值。
    來源：新大腦.pdf - EXT_GM7_Numeric_Gaps_Vec (Page 29-30)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm7-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(
        f"Executing EXT_GM7_Numeric_Gaps_Vec with config: {config.model_dump_json(indent=2)}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores * config.weight # 來源：新大腦.pdf (Page 30)

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 30)
    if not potential_numbers_to_place: return scores * config.weight # 來源：新大腦.pdf (Page 30)
        
    max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols)) #
    if max_board_val == 0: max_board_val = 1.0 # Avoid division by zero issues

    empty_r_indices, empty_c_indices = np.where(grid == -1)

    for r_idx, c_idx in zip(empty_r_indices, empty_c_indices): # 來源：新大腦.pdf (Page 30)
        max_cell_gap_score: float = 0.0 # 來源：新大腦.pdf (Page 30)
        for p_val in potential_numbers_to_place:
            current_max_score_for_pval_direction: float = 0.0
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]: # 來源：新大腦.pdf (Page 30)
                # Case 1: Mend N1 - p_val - N2 (來源：新大腦.pdf Page 30-31)
                r_n1, c_n1 = r_idx - dr, c_idx - dc
                r_n2, c_n2 = r_idx + dr, c_idx + dc
                if (0 <= r_n1 < rows and 0 <= c_n1 < cols and
                    0 <= r_n2 < rows and 0 <= c_n2 < cols):
                    val_n1, val_n2 = grid[r_n1, c_n1], grid[r_n2, c_n2]
                    if val_n1 != -1 and val_n2 != -1:
                        score_update = 0.0
                        if val_n1 == p_val - 1 and val_n2 == p_val + 1: # Arithmetic diff 1 (來源：新大腦.pdf Page 31)
                            score_update = config.score_arithmetic_1_gap_fill
                            if config.enable_quality_enhancement_gm7 and \
                               (val_n1 + p_val + val_n2) / 3.0 > (max_board_val * config.high_value_threshold_factor_gm7):
                                score_update += config.score_gap_fill_high_val_bonus
                        elif (val_n1 + val_n2) == 2 * p_val and not math.isclose(p_val, val_n1): # Generic arithmetic (來源：新大腦.pdf Page 31)
                            score_update = config.score_arithmetic_generic_mend
                        current_max_score_for_pval_direction = max(current_max_score_for_pval_direction, score_update)
                
                # Case 2: Extend p_val - N1 - N2 (來源：新大腦.pdf Page 31)
                r_n1_e1, c_n1_e1 = r_idx + dr, c_idx + dc
                r_n2_e1, c_n2_e1 = r_idx + 2 * dr, c_idx + 2 * dc
                if (0 <= r_n1_e1 < rows and 0 <= c_n1_e1 < cols and
                    0 <= r_n2_e1 < rows and 0 <= c_n2_e1 < cols):
                    val_n1_e1, val_n2_e1 = grid[r_n1_e1, c_n1_e1], grid[r_n2_e1, c_n2_e1]
                    if val_n1_e1 != -1 and val_n2_e1 != -1:
                        common_diff = val_n1_e1 - p_val
                        if not math.isclose(common_diff, 0) and math.isclose(val_n2_e1, val_n1_e1 + common_diff):
                            current_max_score_for_pval_direction = max(current_max_score_for_pval_direction, config.score_arithmetic_generic_extend)

                # Case 3: Extend N1 - N2 - p_val (來源：新大腦.pdf Page 31-32)
                r_n1_e2, c_n1_e2 = r_idx - 2 * dr, c_idx - 2 * dc
                r_n2_e2, c_n2_e2 = r_idx - dr, c_idx - dc
                if (0 <= r_n1_e2 < rows and 0 <= c_n1_e2 < cols and
                    0 <= r_n2_e2 < rows and 0 <= c_n2_e2 < cols):
                    val_n1_e2, val_n2_e2 = grid[r_n1_e2, c_n1_e2], grid[r_n2_e2, c_n2_e2]
                    if val_n1_e2 != -1 and val_n2_e2 != -1:
                        common_diff = val_n2_e2 - val_n1_e2
                        if not math.isclose(common_diff, 0) and math.isclose(p_val, val_n2_e2 + common_diff):
                            current_max_score_for_pval_direction = max(current_max_score_for_pval_direction, config.score_arithmetic_generic_extend)
            max_cell_gap_score = max(max_cell_gap_score, current_max_score_for_pval_direction)
        scores[r_idx, c_idx] = MathUtils.normalize_value(max_cell_gap_score, 0, 1.0, clamp=True) # Scores already ~0-1 + bonus (max ~1.0)
    return scores * config.weight


# 引用：建議.txt (source 654-655, 709-710) - 全面向量化 EXT_GM8
def EXT_GM8_Edge_Affinity_Vec(
    grid: np.ndarray,
    config: EdgeAffinityConfig,
    request_id: str | None = "N/A_GM8_EdgeAff",
) -> np.ndarray:
    """(GM8-邊緣親和度) 來源：新大腦.pdf (Page 31). Vectorized approach."""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm8-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_GM8 with config: {config.model_dump_json(indent=2)}", extra=log_extra)

    rows, cols = grid.shape
    final_scores_matrix = np.zeros((rows, cols), dtype=float) # Initialize for all, then fill for empty
    if rows == 0 or cols == 0: return final_scores_matrix * config.weight

    r_coords, c_coords = np.indices(grid.shape)
    dist_to_top = r_coords
    dist_to_bottom = rows - 1 - r_coords
    dist_to_left = c_coords
    dist_to_right = cols - 1 - c_coords
    min_dist_to_edge = np.minimum.reduce([dist_to_top, dist_to_bottom, dist_to_left, dist_to_right]).astype(float)

    max_min_dist_row = (rows - 1) // 2
    max_min_dist_col = (cols - 1) // 2
    overall_max_of_min_distances = float(min(max_min_dist_row, max_min_dist_col))
    # 引用：知識大典.txt – 防錯字典.txt – "ZeroDivisionError" (防範)
    if overall_max_of_min_distances < 1e-9 : # Covers 0 and very small values
        overall_max_of_min_distances = 0.5 if (rows <=2 or cols <=2) else 1.0 
    
    normalized_dist = np.clip(min_dist_to_edge / overall_max_of_min_distances, 0.0, 1.0)

    cell_scores = np.zeros_like(normalized_dist)
    if config.affinity_mode == "prefer_edge":
        cell_scores = 1.0 - normalized_dist
        is_corner_mask = ((r_coords == 0) | (r_coords == rows - 1)) & ((c_coords == 0) | (c_coords == cols - 1))
        on_edge_mask = (min_dist_to_edge < 1e-9) # Check if effectively on edge
        cell_scores[is_corner_mask & on_edge_mask] += config.corner_bonus_prefer
    elif config.affinity_mode == "avoid_edge":
        cell_scores = normalized_dist
        is_corner_mask = ((r_coords == 0) | (r_coords == rows - 1)) & ((c_coords == 0) | (c_coords == cols - 1))
        on_edge_mask = (min_dist_to_edge < 1e-9)
        cell_scores[is_corner_mask & on_edge_mask] -= config.corner_penalty_avoid
    else: # Should not happen due to Pydantic pattern validation
        cell_scores = np.full(grid.shape, 0.5, dtype=float)

    min_norm_val = -config.corner_penalty_avoid if config.affinity_mode == "avoid_edge" else 0.0
    max_norm_val = 1.0 + config.corner_bonus_prefer if config.affinity_mode == "prefer_edge" else 1.0
    if math.isclose(max_norm_val, min_norm_val): max_norm_val = min_norm_val + 1.0

    # Apply normalization to the calculated cell_scores
    # MathUtils.normalize_value expects scalar or list, need to adapt for array or apply element-wise
    # For array, we can implement normalize_value's logic directly:
    if not math.isclose(max_norm_val, min_norm_val):
        normalized_cell_scores = (cell_scores - min_norm_val) / (max_norm_val - min_norm_val)
    else: # Handle case where range is zero
        normalized_cell_scores = np.full_like(cell_scores, 0.5 if math.isclose(cell_scores[0,0], min_norm_val) else (0.0 if cell_scores[0,0] < min_norm_val else 1.0) )

    clamped_scores = np.clip(normalized_cell_scores, 0.0, 1.0)
    
    empty_mask = (grid == -1)
    final_scores_matrix[empty_mask] = clamped_scores[empty_mask]
    
    return final_scores_matrix * config.weight

# 引用：建議.txt (source 650, 705) - 全面向量化 EXT_GM9
def EXT_GM9_Center_Control_Vec(
    grid: np.ndarray,
    config: CenterControlConfig,
    request_id: str | None = "N/A_GM9_CenterCtrl",
) -> np.ndarray:
    """(GM9-中心控制偏好) 來源：新大腦.pdf (Page 34). Vectorized approach."""
    if not config.enabled: return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm9-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(f"Executing EXT_GM9 with config: {config.model_dump_json(indent=2)}", extra=log_extra)

    rows, cols = grid.shape
    final_scores_matrix = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return final_scores_matrix * config.weight

    r_coords, c_coords = np.indices(grid.shape, dtype=float) # Use float for center calculation
    center_r, center_c = (rows - 1.0) / 2.0, (cols - 1.0) / 2.0
    
    distances_to_center = np.sqrt((r_coords - center_r)**2 + (c_coords - center_c)**2)
    
    max_dist_to_center = MathUtils.euclidean_distance((0.0, 0.0), (center_r, center_c))
    # 引用：知識大典.txt – 防錯字典.txt – "ZeroDivisionError" (防範)
    if max_dist_to_center < 1e-9: # Handles 1x1 grid or cases where center is (0,0) for dist calc
        max_dist_to_center = 1.0 # Avoid division by zero, or provide meaningful scale

    normalized_dist = np.clip(distances_to_center / max_dist_to_center, 0.0, 1.0)

    cell_scores: np.ndarray
    if config.affinity_mode == "prefer_center":
        cell_scores = 1.0 - normalized_dist
    elif config.affinity_mode == "avoid_center":
        cell_scores = normalized_dist
    else: # Should not happen due to Pydantic validation
        cell_scores = np.full(grid.shape, 0.5, dtype=float)
    
    clamped_scores = np.clip(cell_scores, 0.0, 1.0) # Already normalized, but clamp for safety
    
    empty_mask = (grid == -1)
    final_scores_matrix[empty_mask] = clamped_scores[empty_mask]
    
    return final_scores_matrix * config.weight


# 來源：新大腦.pdf - 16. EXT_GM10_Blocking_Value_Vec (Page 35-38)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM10強化建議
def EXT_GM10_Blocking_Value_Vec(
    grid: np.ndarray,
    config: BlockingValueConfig,
    request_id: str | None = "N/A_GM10_Blocking",
) -> np.ndarray:
    """
    (GM10-阻斷價值評估) 評估填補是否避免形成不良模式。
    來源：新大腦.pdf - EXT_GM10_Blocking_Value_Vec (Page 35)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm10-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(
        f"Executing EXT_GM10_Blocking_Value_Vec with config: {config.model_dump_json(indent=2)}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores * config.weight # 來源：新大腦.pdf (Page 36)

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 36)
    if not potential_numbers_to_place: return scores * config.weight # 來源：新大腦.pdf (Page 36)

    UNDESIRABLE_SEQUENCES = [tuple(seq) for seq in config.undesirable_sequences_list if len(seq) == config.check_line_length] # Convert to tuple for set operations if needed
    line_len_to_check = config.check_line_length

    empty_r_indices, empty_c_indices = np.where(grid == -1)

    for r_idx, c_idx in zip(empty_r_indices, empty_c_indices): # 來源：新大腦.pdf (Page 36)
        max_safety_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 36)
        evaluated_any_pval = False
        for p_val in potential_numbers_to_place:
            evaluated_any_pval = True
            temp_grid = grid.copy()
            temp_grid[r_idx, c_idx] = p_val
            forms_undesirable_pattern_for_pval = False # 來源：新大腦.pdf (Page 37)

            for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: # 來源：新大腦.pdf (Page 36)
                if forms_undesirable_pattern_for_pval: break
                for i_offset_in_window in range(line_len_to_check): # 來源：新大腦.pdf (Page 37)
                    current_line_values_list: List[int] = []
                    is_valid_segment = True
                    
                    # Construct the line segment efficiently
                    # 引用：建議.txt (source 651, 706) - 鄰域操作 (雖非直接向量化整個模組，但內部操作可優化)
                    r_coords = r_idx + (np.arange(line_len_to_check) - i_offset_in_window) * dr_line
                    c_coords = c_idx + (np.arange(line_len_to_check) - i_offset_in_window) * dc_line
                    
                    if not (np.all((r_coords >= 0) & (r_coords < rows) & (c_coords >= 0) & (c_coords < cols))):
                        is_valid_segment = False
                    
                    if is_valid_segment:
                        current_line_values_list = [int(temp_grid[r,c]) for r,c in zip(r_coords, c_coords)] # Convert to list of int
                        # 來源：新大腦.pdf (Page 37)
                        if tuple(current_line_values_list) in UNDESIRABLE_SEQUENCES: # Compare tuple to list of tuples
                            forms_undesirable_pattern_for_pval = True
                            break 
                if forms_undesirable_pattern_for_pval: break 
            
            current_score_for_pval = config.score_if_safe if not forms_undesirable_pattern_for_pval else config.score_if_unsafe # 來源：新大腦.pdf (Page 37)
            if current_score_for_pval > max_safety_score_for_cell:
                max_safety_score_for_cell = current_score_for_pval
        
        if not evaluated_any_pval : # Only if potential_numbers_to_place was empty for this cell (should not happen if global list not empty)
             scores[r_idx,c_idx] = 0.5 # Neutral
        else:
             scores[r_idx, c_idx] = max_safety_score_for_cell # 來源：新大腦.pdf (Page 37-38)

    return scores * config.weight


# 來源：新大腦.pdf - 17. EXT_GM11_Pair_Correlation_Vec (Page 38-39)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM11強化建議
def EXT_GM11_Pair_Correlation_Vec(
    grid: np.ndarray,
    config: PairCorrelationConfig,
    request_id: str | None = "N/A_GM11_PairCorr",
) -> np.ndarray:
    """
    (GM11-數字配對關聯分析) 分析特定數字對共同出現的價值。
    來源：新大腦.pdf - EXT_GM11_Pair_Correlation_Vec (Page 38)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm11-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(
        f"Executing EXT_GM11_Pair_Correlation_Vec with config: {config.model_dump_json(indent=2)}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: return scores * config.weight # 來源：新大腦.pdf (Page 38)

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 38)
    if not potential_numbers_to_place: return scores * config.weight # 來源：新大腦.pdf (Page 38)

    # Normalize keys in favorable_pairs for consistent lookup, or ensure config is pre-normalized
    # For this example, assume config.favorable_pairs might have ordered tuples.
    # PDF implies order matters: (p_val, neighbor_val)
    
    max_single_pair_score: float = 0.0 # 來源：新大腦.pdf (Page 38)
    if config.favorable_pairs:
        max_single_pair_score = float(max(config.favorable_pairs.values())) if config.favorable_pairs.values() else 0.0
    
    heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 1e-9 else 1.0 # 來源：新大腦.pdf (Page 39)

    empty_r_indices, empty_c_indices = np.where(grid == -1)

    for r_idx, c_idx in zip(empty_r_indices, empty_c_indices): # 來源：新大腦.pdf (Page 39)
        max_accumulated_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 39)
        for p_val in potential_numbers_to_place:
            current_pval_accumulated_score: float = 0.0
            # 引用：建議.txt (source 651, 706) - 鄰域操作的向量化 (可以獲取整個鄰域窗口)
            # For 8 neighbors directly:
            for dr_n, dc_n in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]: # 來源：新大腦.pdf (Page 39)
                nr, nc = r_idx + dr_n, c_idx + dc_n
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor_val = grid[nr, nc]
                    if neighbor_val != -1: # 來源：新大腦.pdf (Page 39)
                        pair_key = (p_val, int(neighbor_val))
                        current_pval_accumulated_score += config.favorable_pairs.get(pair_key, 0.0) # Use .get for safety
            
            if current_pval_accumulated_score > max_accumulated_score_for_cell: # 來源：新大腦.pdf (Page 39)
                max_accumulated_score_for_cell = current_pval_accumulated_score
            
        scores[r_idx, c_idx] = MathUtils.normalize_value(
            max_accumulated_score_for_cell, 0, heuristic_max_total_pair_score, clamp=True
        ) # 來源：新大腦.pdf (Page 39)
            
    return scores * config.weight


# 來源：新大腦.pdf - 18. EXT_GM12_Island_Analysis_Vec (Page 39-41)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM12強化建議
# BFS is inherently iterative, vectorization of BFS core is not straightforward with NumPy.
# Optimizations: efficient queue, NumPy for post-BFS calculations.
def EXT_GM12_Island_Analysis_Vec(
    grid: np.ndarray,
    config: IslandAnalysisConfig,
    request_id: str | None = "N/A_GM12_Island",
) -> np.ndarray:
    """
    (GM12 - 島嶼分析) 分析已填數字形成的「島嶼」特性。
    來源：新大腦.pdf - EXT_GM12_Island_Analysis_Vec (Page 39-40)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm12-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(
        f"Executing EXT_GM12_Island_Analysis_Vec with config: {config.model_dump_json(indent=2)}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) # Empty cells get 0 for this module
    if rows == 0 or cols == 0: return scores * config.weight # 來源：新大腦.pdf (Page 40)

    visited_island_search = np.zeros_like(grid, dtype=bool) # 來源：新大腦.pdf (Page 40)
    max_val_on_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 40)
    if max_val_on_board == 0: max_val_on_board = 1.0 # 來源：新大腦.pdf (Page 40)

    w_size, w_compactness, w_avg_value = config.w_size, config.w_compactness, config.w_avg_value # 來源：新大腦.pdf (Page 40)

    # Iterate to find starting points of islands
    # 引用：建議.txt (source 650, 705) - 雖然BFS本身迭代，但周邊可用NumPy
    # Find all potential island start cells (filled and not visited)
    # This can be done more efficiently than iterating every cell in Python.
    potential_starts_r, potential_starts_c = np.where((grid != -1) & (~visited_island_search))

    for r_start, c_start in zip(potential_starts_r, potential_starts_c):
        if visited_island_search[r_start, c_start]: # Already processed as part of another island
            continue

        current_island_cells: List[Tuple[int, int]] = [] # 來源：新大腦.pdf (Page 40)
        current_island_values: List[int] = [] # 來源：新大腦.pdf (Page 40)
        
        q = deque([(r_start, c_start)]) # 來源：新大腦.pdf (Page 40)
        visited_island_search[r_start, c_start] = True
        
        # Bounding box for compactness
        # 來源：新大腦.pdf (Page 40)
        min_r_bbox, max_r_bbox = r_start, r_start 
        min_c_bbox, max_c_bbox = c_start, c_start 

        while q: # 來源：新大腦.pdf (Page 40)
            r_curr, c_curr = q.popleft()
            current_island_cells.append((r_curr, c_curr))
            current_island_values.append(int(grid[r_curr, c_curr]))

            min_r_bbox = min(min_r_bbox, r_curr); max_r_bbox = max(max_r_bbox, r_curr) # 來源：新大腦.pdf (Page 40-41)
            min_c_bbox = min(min_c_bbox, c_curr); max_c_bbox = max(max_c_bbox, c_curr) # 來源：新大腦.pdf (Page 40-41)

            for dr_bfs, dc_bfs in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-connectivity (來源：新大腦.pdf Page 41)
                nr, nc = r_curr + dr_bfs, c_curr + dc_bfs
                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr, nc] != -1 and not visited_island_search[nr, nc]): # 來源：新大腦.pdf (Page 41)
                    visited_island_search[nr, nc] = True
                    q.append((nr, nc))
        
        if not current_island_cells: continue # Should not happen if starting from a filled cell

        island_size = float(len(current_island_cells)) # 來源：新大腦.pdf (Page 41)
        avg_value_island = np.mean(current_island_values) if current_island_values else 0.0 # Use np.mean
        
        bbox_height = float(max_r_bbox - min_r_bbox + 1)
        bbox_width = float(max_c_bbox - min_c_bbox + 1)
        bbox_area = bbox_height * bbox_width
        compactness = island_size / bbox_area if bbox_area > 1e-9 else 0.0 # 來源：新大腦.pdf (Page 41)
        
        norm_size = MathUtils.normalize_value(island_size, 1, float(rows * cols), clamp=True) # 來源：新大腦.pdf (Page 41)
        norm_compactness = MathUtils.normalize_value(compactness, 0, 1.0, clamp=True) # Compactness is already 0-1
        norm_avg_value = MathUtils.normalize_value(avg_value_island, 1, max_val_on_board, clamp=True) # 來源：新大腦.pdf (Page 41)

        total_weights = w_size + w_compactness + w_avg_value # 來源：新大腦.pdf (Page 41)
        # 引用：知識大典.txt – 防錯字典.txt – "ZeroDivisionError" (防範)
        if total_weights < 1e-9 : total_weights = 1.0 

        island_score_unnormalized = (w_size * norm_size + w_compactness * norm_compactness + w_avg_value * norm_avg_value) #
        final_island_score = MathUtils.normalize_value(island_score_unnormalized, 0, total_weights, clamp=True) # Normalize against sum of weights

        for r_cell, c_cell in current_island_cells: # Assign score to all cells in this island (來源：新大腦.pdf Page 41)
            scores[r_cell, c_cell] = final_island_score
            
    return scores * config.weight


# === Appended Registry ===
"""
Module registry for brain modules.
Automatically generated to support API integration.
"""

from typing import Dict, Type
from brain1 import *
from brain2 import *
from brain3 import *

DEFAULT_MODULE_CONFIGS: Dict[str, Type] = {
    "basemodule": BaseModuleConfig,
    "weightedproximity": WeightedProximityConfig,
    "localheterogeneity": LocalHeterogeneityConfig,
    "potentialfield": PotentialFieldConfig,
    "discontinuityrepair": DiscontinuityRepairConfig,
    "pathfindingvalue": PathfindingValueConfig,
    "resourcecontrol": ResourceControlConfig,
    "linecontrol": LineControlConfig,
    "connectedcomponent": ConnectedComponentConfig,
    "spatialautocorrelation": SpatialAutocorrelationConfig,
    "linecompletion": LineCompletionConfig,
    "symmetrypotential": SymmetryPotentialConfig,
    "numericgaps": NumericGapsConfig,
    "edgeaffinity": EdgeAffinityConfig,
    "centercontrol": CenterControlConfig,
    "blockingvalue": BlockingValueConfig,
    "paircorrelation": PairCorrelationConfig,
    "islandanalysis": IslandAnalysisConfig,
    "sequencediversity": SequenceDiversityConfig,
    "riskassessment": RiskAssessmentConfig,
    "informationgain": InformationGainConfig,
    "harmoniccentrality": HarmonicCentralityConfig,
    "localentropyminimization": LocalEntropyMinimizationConfig,
    "rlvalueestimation": RLValueEstimationConfig,
    "skippattern": SkipPatternConfig,
    "skippatternconfidence": SkipPatternConfidenceConfig
}

REGISTERED_MODULES_BRAIN: Dict[str, str] = {
    "basemodule": "brain1.BaseModuleConfig",
    "weightedproximity": "brain1.WeightedProximityConfig",
    "localheterogeneity": "brain1.LocalHeterogeneityConfig",
    "potentialfield": "brain1.PotentialFieldConfig",
    "discontinuityrepair": "brain1.DiscontinuityRepairConfig",
    "pathfindingvalue": "brain1.PathfindingValueConfig",
    "resourcecontrol": "brain1.ResourceControlConfig",
    "linecontrol": "brain1.LineControlConfig",
    "connectedcomponent": "brain1.ConnectedComponentConfig",
    "spatialautocorrelation": "brain2.SpatialAutocorrelationConfig",
    "linecompletion": "brain2.LineCompletionConfig",
    "symmetrypotential": "brain2.SymmetryPotentialConfig",
    "numericgaps": "brain2.NumericGapsConfig",
    "edgeaffinity": "brain2.EdgeAffinityConfig",
    "centercontrol": "brain2.CenterControlConfig",
    "blockingvalue": "brain2.BlockingValueConfig",
    "paircorrelation": "brain2.PairCorrelationConfig",
    "islandanalysis": "brain2.IslandAnalysisConfig",
    "sequencediversity": "brain3.SequenceDiversityConfig",
    "riskassessment": "brain3.RiskAssessmentConfig",
    "informationgain": "brain3.InformationGainConfig",
    "harmoniccentrality": "brain3.HarmonicCentralityConfig",
    "localentropyminimization": "brain3.LocalEntropyMinimizationConfig",
    "rlvalueestimation": "brain3.RLValueEstimationConfig",
    "skippattern": "brain3.SkipPatternConfig",
    "skippatternconfidence": "brain3.SkipPatternConfidenceConfig"
}



# === Injected Function ===

def get_module_score(config, board_state) -> float:
    """
    統一模組分數計算介面。
    """
    return 0.0  # TODO: 根據實際模組策略實作
