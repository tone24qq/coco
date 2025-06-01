# brain2.py
# Part 2 of 3: Contains the second set of AI scoring modules.
# Based on Brain.txt, which was generated according to 新大腦.pdf, 给你2025资料在深度建议一次.pdf, 极限强化.pdf

import numpy as np
import math
from collections import Counter, deque # deque might not be used here, Counter might be
import logging
from typing import List, Dict, Tuple, Callable, Optional, Any, Set

from pydantic import BaseModel, Field

# Assuming brain1.py is in the same path and contains these definitions
from brain1 import BaseModuleConfig, MathUtils, BoardAnalyzerUtils

logger = logging.getLogger(__name__)

# --- Pydantic Config Models for Modules (Continued from brain1.py) ---

class SpatialAutocorrelationConfig(BaseModuleConfig): # For GM4
    # 來源：新大腦.pdf - EXT_GM4 parameters (Page 23-24)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM4 自相關性方向 #
    autocorrelation_type: str = Field(default="positive", pattern="^(positive|negative)$", description="偏好的自相關類型（positive: 聚集, negative: 交錯）")
    neighborhood_radius: int = Field(default=1, ge=1)
    use_median_for_hypothetical: bool = Field(default=True, description="是否使用潛在數字的中位數作為假設值，否則用平均值")


class LineCompletionConfig(BaseModuleConfig): # For GM5
    # 來源：新大腦.pdf - EXT_GM5 parameters (Page 24-25)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM5 線段長度和類型的擴展 #
    target_line_length: int = Field(default=3, ge=3, description="目標補全的線段長度")
    score_identical_3: float = Field(default=0.6, ge=0.0)
    score_arithmetic_3_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_3_extend: float = Field(default=0.5, ge=0.0)
    # 來源：新大腦.pdf - EXT_GM5 Added: scoring for quality (conceptual) (Page 25)
    enable_quality_enhancement: bool = Field(default=True) #
    score_arithmetic_3_mend_high_val_bonus: float = Field(default=0.2, ge=0.0, description="高價值等差序列修復額外獎勵") # PDF uses 0.9 directly, here use as bonus
    high_value_threshold_factor_gm5: float = Field(default=0.66, ge=0, le=1, description="平均值超過盤面最大值*此因子視為高價值")


class SymmetryPotentialConfig(BaseModuleConfig): # For GM6
    # 來源：新大腦.pdf - EXT_GM6 parameters (Page 27-28)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM6 對稱類型權重 #
    score_horizontal: float = Field(default=0.7, ge=0.0)
    score_vertical: float = Field(default=0.7, ge=0.0)
    score_point_center: float = Field(default=0.8, ge=0.0)
    score_main_diagonal: float = Field(default=0.6, ge=0.0)
    score_anti_diagonal: float = Field(default=0.6, ge=0.0)
    strict_square_for_diagonal: bool = Field(default=True, description="對角線對稱是否嚴格要求方形棋盤") # 來源：新大腦.pdf (Page 29) #


class NumericGapsConfig(BaseModuleConfig): # For GM7
    # 來源：新大腦.pdf - EXT_GM7 parameters (Page 29-30)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM7 「間隙」的上下文 #
    score_arithmetic_1_gap_fill: float = Field(default=0.9, ge=0.0)
    score_arithmetic_generic_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_generic_extend: float = Field(default=0.5, ge=0.0)
    # 來源：新大腦.pdf - EXT_GM7 Added: scoring for quality (conceptual) (Page 30)
    enable_quality_enhancement_gm7: bool = Field(default=True) #
    score_gap_fill_high_val_bonus: float = Field(default=0.1, ge=0.0) # PDF uses 0.95 directly
    high_value_threshold_factor_gm7: float = Field(default=0.66, ge=0, le=1)
    # Conceptual: score_gap_fill_long_seq_potential


class EdgeAffinityConfig(BaseModuleConfig): # For GM8
    # 來源：新大腦.pdf - EXT_GM8 parameters (Page 31-32)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM8 affinity_mode 的動態化 #
    affinity_mode: str = Field(default="prefer_edge", pattern="^(prefer_edge|avoid_edge)$") #
    corner_bonus_prefer: float = Field(default=0.2, ge=0.0) #
    corner_penalty_avoid: float = Field(default=0.2, ge=0.0) #


class CenterControlConfig(BaseModuleConfig): # For GM9
    # 來源：新大腦.pdf - EXT_GM9 parameters (Page 34) #
    affinity_mode: str = Field(default="prefer_center", pattern="^(prefer_center|avoid_center)$") #

# Config for GM10 (BlockingValueConfig) - Using the updated definition from Brain.txt (source 408-410)
class BlockingValueConfig(BaseModuleConfig):
    # 來源：新大腦.pdf - EXT_GM10 parameters (Page 35-36)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM10 UNDESIRABLE_SEQUENCES 的擴展與學習
    undesirable_sequences_list: List[List[int]] = Field(default_factory=lambda: [ #
        [1, 1, 1], [2, 2, 2] # 來源：新大腦.pdf (Page 36)
        # Example: [1, 2, 3] if bad in some contexts
    ])
    # 來源：新大腦.pdf - EXT_GM10 Score logic (Page 37)
    # PDF uses 0.9 if not forms_undesirable, 0.1 if forms.
    # Let's make these configurable.
    score_if_safe: float = Field(default=0.9, ge=0.0, le=1.0, description="Score if placement does NOT complete an undesirable pattern.") #
    score_if_unsafe: float = Field(default=0.1, ge=0.0, le=1.0, description="Score if placement DOES complete an undesirable pattern.")
    check_line_length: int = Field(default=3, ge=2, description="Length of lines to check for undesirable patterns.")


class PairCorrelationConfig(BaseModuleConfig): # For GM11
    # 來源：新大腦.pdf - EXT_GM11 parameters (Page 38-39)
    # FAVORABLE_PAIRS_SCORES can be complex, for now, allow defining a few key ones in config
    # A more advanced config might load these from a file or a larger structure. #
    favorable_pairs: Dict[Tuple[int, int], float] = Field(default_factory=lambda: {
        (3, 7): 0.8, (7, 3): 0.8, (1, 2): 0.6, (2, 1): 0.6, (10,20):0.7, (20,10):0.7
    })


class IslandAnalysisConfig(BaseModuleConfig): # For GM12
    # 來源：新大腦.pdf - EXT_GM12 parameters (Page 40-41)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM12 島嶼特徵的豐富化 #
    w_size: float = Field(default=0.4, ge=0.0, le=1.0) #
    w_compactness: float = Field(default=0.3, ge=0.0, le=1.0) #
    w_avg_value: float = Field(default=0.3, ge=0.0, le=1.0) #
    # Conceptual: add w_shape_factor, w_boundary_value etc.


# --- Scoring Module Implementations (Continued) ---

# 來源：新大腦.pdf - 10. EXT_GM4_Spatial_Auto_Corr_Vec (Page 23)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM4強化建議
# Config for this (SpatialAutocorrelationConfig) was defined in PART 2
def EXT_GM4_Spatial_Auto_Corr_Vec( #
    grid: np.ndarray,
    config: SpatialAutocorrelationConfig,
    request_id: str | None = "N/A_GM4_SpatialAutoCorr", #
) -> np.ndarray:
    """
    (GM4 - 空間自相關性分析)
    核心規則:評估在空格填入一個假設的「平均」潛在數字後,該數字與其周圍現有數字的相似程度。
    目的:鼓勵形成數值聚集(正自相關)或數值交錯(負自相關,但此處偏好正自相關)。此版本偏好正自相關,即填入的數字與周圍鄰居的平均值相似時得分較高。
    啟發式類型:空間統計
    輸出詮釋:分數越高表示填入一個「典型」數字後,能更好地融入周圍環境,形成數值上的聚集。
    來源：新大腦.pdf - EXT_GM4_Spatial_Auto_Corr_Vec (Page 23)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM4"
    logger.debug(
        f"Executing EXT_GM4_Spatial_Auto_Corr_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape #
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0:
        return scores

    potential_numbers = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 23)

    hypothetical_val_to_place: float # 來源：新大腦.pdf (Page 23)
    if potential_numbers:
        if config.use_median_for_hypothetical:
            hypothetical_val_to_place = float(np.median(potential_numbers))
        else:
            hypothetical_val_to_place = float(np.mean(potential_numbers)) #
    else:
        # 來源：新大腦.pdf - EXT_GM4 Fallback for hypothetical_val_to_place (Page 23-24)
        max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))
        hypothetical_val_to_place = (1.0 + float(max_board_val)) / 2.0 if max_board_val > 0 else 0.5

    max_val_on_grid_for_norm = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols))) # 來源：新大腦.pdf (Page 24)
    if max_val_on_grid_for_norm == 0: max_val_on_grid_for_norm = 1.0 # 來源：新大腦.pdf (Page 24)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells #
                continue

            # Get actual numeric neighbors (non -1)
            # 來源：新大腦.pdf - EXT_GM4 get_neighborhood_values (Page 24)
            neighbor_values = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx,  #
                radius=config.neighborhood_radius, # Use config
                eight_connectivity=True,
                val_func=lambda x: float(x) if x != -1 else None,
                include_center=False
            )

            if not neighbor_values: # 來源：新大腦.pdf (Page 24) #
                scores[r_idx, c_idx] = 0.5  # Neutral score if no neighbors to compare with
                continue

            mean_neighbors = np.mean(neighbor_values) # 來源：新大腦.pdf (Page 24)

            # Calculate the difference between the hypothetical placed value and the mean of its actual neighbors
            # 來源：新大腦.pdf (Page 24) #
            diff_hypothetical_to_mean_neighbors = abs(hypothetical_val_to_place - mean_neighbors)

            # Normalize this difference. Max possible difference is roughly max_val_on_grid.
            # Score for positive autocorrelation: 1.0 - normalized_difference
            # (smaller difference means more similar, thus higher positive autocorrelation score)
            # 來源：新大腦.pdf (Page 24)
            norm_diff = MathUtils.normalize_value(
                diff_hypothetical_to_mean_neighbors, 0, max_val_on_grid_for_norm, clamp=True
            )
            
            current_score: float
            if config.autocorrelation_type == "positive":
                current_score = 1.0 - norm_diff # 來源：新大腦.pdf (Page 24)
            else: # "negative" autocorrelation (交錯)
                current_score = norm_diff # Larger difference is better
            
            scores[r_idx, c_idx] = current_score #
            
    return scores * config.weight


# 來源：新大腦.pdf - 11. EXT_GM5_Line_Completion_Vec (Page 24)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM5強化建議
# Config for this (LineCompletionConfig) was defined in PART 2
def EXT_GM5_Line_Completion_Vec(
    grid: np.ndarray,
    config: LineCompletionConfig,
    request_id: str | None = "N/A_GM5_LineComp", #
) -> np.ndarray:
    """
    (GM5-線段補全)
    核心規則:評估空格對於完成特定方向(行、列、對角線)上具有特定構成(如等差、等值)的短線段(例如長度為3)之潛力。
    目的:偏好那些能夠「臨門一腳」完成有意義短線段的空格。
    啟發式類型:模式匹配(短線段)
    輸出詮釋:分數越高表示該空格填入某數字後,越能完成一個預定義的短線段模式。
    來源：新大腦.pdf - EXT_GM5_Line_Completion_Vec (Page 24)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM5"
    logger.debug(
        f"Executing EXT_GM5_Line_Completion_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape #
    scores = np.zeros((rows, cols), dtype=float)
    # 來源：新大腦.pdf - EXT_GM5 initial checks (Page 25)
    if rows == 0 or cols == 0 or min(rows,cols) < 1: # PDF: min(rows,cols) < 1. For lines of 3, need more.
        # For target_line_length, need at least that many in one dimension.
        # if config.target_line_length > max(rows, cols) and (rows > 0 and cols > 0) : # if grid is smaller than target line
            pass # allow, but scores will likely be 0
    if rows == 0 or cols == 0 or min(rows, cols) < 2:
        return scores


    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid))  # OK

if rows == 0 or cols == 0 or min(rows, cols) < 2:
    return scores  # <- 正確縮排

    # 來源：新大腦.pdf - EXT_GM5 line_completion_score_map (Page 25) #
    # Using config for scores
    
    max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 25)
                continue

            max_completion_score_for_cell: float = 0.0 #

            for p_val in potential_numbers_to_place:
                current_pval_max_score_contribution: float = 0.0
                
                # Directions: Horizontal, Vertical, Diagonal (top-left to bottom-right), Anti-Diagonal
                # 來源：新大腦.pdf - EXT_GM5 Directions (Page 25)
                # Each direction vector (dr, dc) #
                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # For a line of target_line_length, p_val can be at any position within it.
                    # Iterate through all possible windows of target_line_length that include (r_idx, c_idx)
                    # where (r_idx, c_idx) is filled with p_val.
                    for i_offset in range(config.target_line_length): # p_val is at index i_offset in the window #
                        # Start of window relative to (r_idx, c_idx) as if it's the 0-th element in the window
                        # Window cells are: (r_idx + (k-i_offset)*dr_dir, c_idx + (k-i_offset)*dc_dir) for k in 0..L-1
                        
                        current_line_values: List[int] = [] #
                        is_valid_line_segment = True
                        
                        for k_in_segment in range(config.target_line_length): #
                            # Actual coordinates of the k_in_segment-th cell in the current line
                            eval_r = r_idx + (k_in_segment - i_offset) * dr_dir
                            eval_c = c_idx + (k_in_segment - i_offset) * dc_dir #

                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                is_valid_line_segment = False #
                                break
                            
                            if eval_r == r_idx and eval_c == c_idx:
                                current_line_values.append(p_val) #
                            else:
                                current_line_values.append(int(grid[eval_r, eval_c])) # Cast to int if not -1
                        
                        if is_valid_line_segment and all(val != -1 for val in current_line_values): # All cells in segment must be filled #
                            s = current_line_values
                            temp_score_for_this_line = 0.0 #

                            # Check for 3 identical (or target_line_length identical)
                            # 來源：新大腦.pdf - EXT_GM5 Identical 3 Check (Page 26)
                            if len(set(s)) == 1: # All elements are same #
                                temp_score_for_this_line = max(temp_score_for_this_line, config.score_identical_3)
                            
                            # Check for arithmetic (non-constant) #
                            # 來源：新大腦.pdf - EXT_GM5 Arithmetic 3 Mend/Extend (Page 26)
                            # This general check is for a complete line s of target_line_length
                            if len(s) >= 2: #
                                diffs = [s[k+1] - s[k] for k in range(len(s)-1)]
                                if len(set(diffs)) == 1 and diffs[0] != 0: # Is arithmetic and non-constant #
                                    # Determine if it's a "mend" or "extend" based on p_val's position (i_offset)
                                    # This distinction is complex for generic target_line_length. #
                                    # PDF has specific logic for length 3.
                                    if config.target_line_length == 3:
                                        if i_offset == 1: # p_val is in the middle (mending) #
                                            temp_score_for_this_line = max(temp_score_for_this_line, config.score_arithmetic_3_mend)
                                            # 來源：新大腦.pdf - EXT_GM5 Quality Enhancement (Conceptual) (Page 26) #
                                            if config.enable_quality_enhancement:
                                                avg_val_line = sum(s) / len(s)
                                                if max_board_val > 0 and avg_val_line > (max_board_val * config.high_value_threshold_factor_gm5): #
                                                    temp_score_for_this_line += config.score_arithmetic_3_mend_high_val_bonus #
                                        else: # p_val is at an end (extending)
                                            temp_score_for_this_line = max(temp_score_for_this_line, config.score_arithmetic_3_extend) #
                                    else: # For other lengths, use a generic arithmetic score
                                        temp_score_for_this_line = max(temp_score_for_this_line, config.score_arithmetic_3_mend) # Use mend score as base

                            current_pval_max_score_contribution = max(current_pval_max_score_contribution, temp_score_for_this_line) #
                
                max_completion_score_for_cell = max(max_completion_score_for_cell, current_pval_max_score_contribution)

            # Normalize based on the max possible score from config map (approx 1.0 as scores are defined in 0-1 range)
            # 來源：新大腦.pdf - EXT_GM5 Normalization (Page 27) #
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_completion_score_for_cell, 0, 1.0, clamp=True # Max score from map is < 1
            )
            
    return scores * config.weight


# 來源：新大腦.pdf - 12. EXT_GM6_Symmetry_Potential_Vec (Page 27)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM6強化建議
# Config for this (SymmetryPotentialConfig) was defined in PART 2
def EXT_GM6_Symmetry_Potential_Vec( #
    grid: np.ndarray,
    config: SymmetryPotentialConfig,
    request_id: str | None = "N/A_GM6_Symmetry", #
) -> np.ndarray:
    """
    (GM6-對稱性潛力)
    核心規則:評估在空格填入數字後,盤面形成的對稱性程度(水平、垂直、中心、主對角線、反主對角線)。
    目的:偏好那些能夠創造或增強盤面對稱性的填補。
    啟發式類型:幾何與模式識別
    輸出詮釋:分數越高表示若在該空格填入特定數字,能與對稱位置上已存在的相同數字形成對稱。
    來源：新大腦.pdf - EXT_GM6_Symmetry_Potential_Vec (Page 27)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM6"
    logger.debug(
        f"Executing EXT_GM6_Symmetry_Potential_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape #
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 27)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 27)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 28)
        return scores

    # 來源：新大腦.pdf - EXT_GM6 symmetry_scores_map (Page 28) & Conceptual dynamic adjustment
    # Using scores from config directly.
    # Dynamic adjustment idea:
    # score_main_diag = config.score_main_diagonal
    # score_anti_diag = config.score_anti_diagonal
    # if rows == cols and config.strict_square_for_diagonal: # or simply if rows == cols
    #     score_main_diag = max(score_main_diag, 0.7) # Example boost for square grids
    #     score_anti_diag = max(score_anti_diag, 0.7)


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells #
                continue
            
            max_symmetry_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 28)

            for p_val in potential_numbers_to_place:
                current_pval_max_sym: float = 0.0

                # 1. Horizontal Symmetry: (r_idx, c_idx) vs (r_idx, cols - 1 - c_idx) #
                # 來源：新大腦.pdf - Horizontal Symmetry (Page 28)
                sr_h, sc_h = r_idx, cols - 1 - c_idx
                if sc_h != c_idx: # Not the same cell
                    if 0 <= sr_h < rows and 0 <= sc_h < cols and grid[sr_h, sc_h] == p_val: #
                        current_pval_max_sym = max(current_pval_max_sym, config.score_horizontal)

                # 2. Vertical Symmetry: (r_idx, c_idx) vs (rows - 1 - r_idx, c_idx)
                # 來源：新大腦.pdf - Vertical Symmetry (Page 28)
                sr_v, sc_v = rows - 1 - r_idx, c_idx #
                if sr_v != r_idx: # Not the same cell
                    if 0 <= sr_v < rows and 0 <= sc_v < cols and grid[sr_v, sc_v] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, config.score_vertical) #
                
                # 3. Point (Center) Symmetry: (r_idx, c_idx) vs (rows - 1 - r_idx, cols - 1 - c_idx)
                # 來源：新大腦.pdf - Point Center Symmetry (Page 28)
                sr_p, sc_p = rows - 1 - r_idx, cols - 1 - c_idx #
                if sr_p != r_idx or sc_p != c_idx: # Not the same cell
                     if 0 <= sr_p < rows and 0 <= sc_p < cols and grid[sr_p, sc_p] == p_val:
                        current_pval_max_sym = max(current_pval_max_sym, config.score_point_center)

                # 4. Main Diagonal Symmetry (\): (r_idx, c_idx) vs (c_idx, r_idx) #
                # 來源：新大腦.pdf - Main Diagonal Symmetry (Page 28-29)
                if not config.strict_square_for_diagonal or rows == cols: # 來源：新大腦.pdf (Page 29)
                    sr_d1, sc_d1 = c_idx, r_idx
                    if sr_d1 != r_idx or sc_d1 != c_idx: # Not the same cell (only if r_idx != c_idx) #
                        if 0 <= sr_d1 < rows and 0 <= sc_d1 < cols and grid[sr_d1, sc_d1] == p_val:
                            current_pval_max_sym = max(current_pval_max_sym, config.score_main_diagonal) #
                
                # 5. Anti-Diagonal Symmetry (/): (r_idx, c_idx) vs ( (cols-1)-c_idx, (rows-1)-r_idx ) - This definition seems more standard for matrix anti-diagonal
                # PDF text for anti-diagonal (Page 29) has discussion.
                # "grid[ (cols-1)-c , (rows-1)-r]" - assuming (cols-1) is max_col_idx, (rows-1) is max_row_idx
                # So if grid[r_original, c_original], symmetric is grid[max_row_idx - c_original_reflected_to_row, max_col_idx - r_original_reflected_to_col]
                # More directly: For grid[r,c] on NxM grid, the anti-diagonal reflection is often considered grid[M-1-c, N-1-r] if axes were swapped.
                # The PDF has: sr_d2, sc_d2 = (cols - 1) - c_idx, (rows - 1) - r_idx - This implies swapping indices AND reflecting.
                # Let's use the PDF's direct formula: (r,c) vs ( (cols-1)-c, (rows-1)-r ) assuming it means the element at row=(cols-1)-c_idx, col=(rows-1)-r_idx
                # This makes sense if you imagine rotating the coordinate system.
                # Let's use a common definition for anti-diagonal element for grid[r_idx][c_idx] in an Rows x Cols grid:
                # The element anti-diagonally symmetric to (r, c) is ( (Rows-1) - (c_prime), (Cols-1) - (r_prime) ) where (r_prime,c_prime) are indices on a transposed conceptual grid.
                # A simpler interpretation often used: if you reflect along the line y = -x + (N-1) for an NxN matrix,
                # (r,c) maps to (N-1-c, N-1-r).
                # We will use this if rows==cols.
                # 來源：新大腦.pdf - Anti-Diagonal Symmetry (Page 29)
                if not config.strict_square_for_diagonal or rows == cols: # 來源：新大腦.pdf (Page 29)
                    # Using (N-1-c, N-1-r) for square N x N (where N=rows=cols)
                    # For general Rows x Cols, this type of symmetry is less strictly defined. #
                    # The PDF uses sr_d2, sc_d2 = (cols - 1) - c_idx, (rows - 1) - r_idx.
                    # This formula assumes grid indices can be derived this way.
                    # If rows=3, cols=5. For (0,0), this gives (4,2).
                    # For (0,4), this gives (0,2).
                    # This seems to be a specific definition of anti-diagonal symmetry. Let's use it.
                    sr_d2, sc_d2 = (rows - 1) - c_idx, (cols - 1) - r_idx # Corrected based on common understanding for anti-diagonal in matrix (N-1-j, M-1-i) #
                    if (sr_d2 != r_idx or sc_d2 != c_idx): # Not the same cell
                        if 0 <= sr_d2 < rows and 0 <= sc_d2 < cols and grid[sr_d2, sc_d2] == p_val:
                            current_pval_max_sym = max(current_pval_max_sym, config.score_anti_diagonal) #

                if current_pval_max_sym > max_symmetry_score_for_cell: # 來源：新大腦.pdf (Page 29)
                    max_symmetry_score_for_cell = current_pval_max_sym
            
            # Scores are already ~0-1 from config map #
            # 來源：新大腦.pdf - EXT_GM6 Normalize (Page 29)
            scores[r_idx, c_idx] = MathUtils.normalize_value(max_symmetry_score_for_cell, 0, 1.0, clamp=True) 
                                                        # Max of map is 0.8 in PDF example, so 1.0 is safe upper for norm. #
    return scores * config.weight

# 來源：新大腦.pdf - 13. EXT_GM7_Numeric_Gaps_Vec (Page 29)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM7強化建議
# Config for this (NumericGapsConfig) was defined in PART 2
def EXT_GM7_Numeric_Gaps_Vec(
    grid: np.ndarray,
    config: NumericGapsConfig,
    request_id: str | None = "N/A_GM7_NumGaps",
) -> np.ndarray:
    """
    (GM7 - 數值間隙填充)
    核心規則:識別並評估在局部區域或序列中,填補數字「間隙」的價值。特別是尋找能填入使之成為公差為1的連續數列的間隙。
    目的:偏好那些能填補序列中明顯缺失數字的空格。
    啟發式類型:序列與模式識別(間隙填充)
    輸出詮釋:分數越高表示該空格若填入特定數字,越能完美地填補一個數值間隙(尤其是公差為1的序列)。 #
    來源：新大腦.pdf - EXT_GM7_Numeric_Gaps_Vec (Page 29-30)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM7"
    logger.debug(
        f"Executing EXT_GM7_Numeric_Gaps_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 30)
        return scores #

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 30)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 30)
        return scores
        
    max_board_val = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows,cols))

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 30)
                continue #
            
            max_cell_gap_score: float = 0.0 # 來源：新大腦.pdf (Page 30)

            for p_val in potential_numbers_to_place:
                # current_pval_score: float = 0.0 # PDF seems to use max_cell_gap_score directly updated
                
                # Iterate over 4 directions (Horizontal, Vertical, Main Diagonal, Anti-Diagonal) #
                # 來源：新大腦.pdf - EXT_GM7 Directions (Page 30)
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    # Case 1: p_val mends a gap: N1 - p_val - N2 #
                    # 來源：新大腦.pdf - EXT_GM7 Case 1 (Page 30)
                    r_n1, c_n1 = r_idx - dr, c_idx - dc
                    r_n2, c_n2 = r_idx + dr, c_idx + dc

                    if 0 <= r_n1 < rows and 0 <= c_n1 < cols and \
                       0 <= r_n2 < rows and 0 <= c_n2 < cols: #
                        val_n1 = grid[r_n1, c_n1]
                        val_n2 = grid[r_n2, c_n2] #

                        if val_n1 != -1 and val_n2 != -1: # Both neighbors exist
                            # Specific check for arithmetic sequence with common difference 1
                            # 來源：新大腦.pdf - EXT_GM7 arithmetic_1_gap_fill (Page 31) #
                            if val_n1 == p_val - 1 and val_n2 == p_val + 1:
                                score = config.score_arithmetic_1_gap_fill
                                # 來源：新大腦.pdf - EXT_GM7 Quality Enhancement (Conceptual) (Page 31) #
                                if config.enable_quality_enhancement_gm7:
                                     if max_board_val > 0 and (val_n1 + p_val + val_n2) / 3.0 > (max_board_val * config.high_value_threshold_factor_gm7): #
                                        score += config.score_gap_fill_high_val_bonus # Add bonus
                                max_cell_gap_score = max(max_cell_gap_score, score)
                            
                            # Generic arithmetic sequence check (d!=0) #
                            # 來源：新大腦.pdf - EXT_GM7 arithmetic_generic_mend (Page 31)
                            elif (val_n1 + val_n2) == 2 * p_val and abs(p_val - val_n1) > 1e-6 : # Not constant, use tolerance for float p_val #
                                max_cell_gap_score = max(max_cell_gap_score, config.score_arithmetic_generic_mend)

                    # Case 2: p_val extends a sequence: p_val - N1 - N2 #
                    # 來源：新大腦.pdf - EXT_GM7 Case 2 (Page 31)
                    r_n1_ext1, c_n1_ext1 = r_idx + dr, c_idx + dc
                    r_n2_ext1, c_n2_ext1 = r_idx + 2 * dr, c_idx + 2 * dc
                    
                    if 0 <= r_n1_ext1 < rows and 0 <= c_n1_ext1 < cols and \
                       0 <= r_n2_ext1 < rows and 0 <= c_n2_ext1 < cols: #
                        val_n1_ext1 = grid[r_n1_ext1, c_n1_ext1] #
                        val_n2_ext1 = grid[r_n2_ext1, c_n2_ext1]

                        if val_n1_ext1 != -1 and val_n2_ext1 != -1:
                            # Check for N1=p_val+d, N2=p_val+2d  => val_n1_ext1 - p_val == val_n2_ext1 - val_n1_ext1 (d) #
                            # 來源：新大腦.pdf - EXT_GM7 Case 2 logic (Page 31)
                            # The PDF has `common_diff = val_n1_ext1 - p_val`
                            # `if common_diff !=0 and val_n2_ext1 == val_n1_ext1 + common_diff:` #
                            common_diff = val_n1_ext1 - p_val
                            if not math.isclose(common_diff, 0) and math.isclose(val_n2_ext1, val_n1_ext1 + common_diff):
                                max_cell_gap_score = max(max_cell_gap_score, config.score_arithmetic_generic_extend) #
                    
                    # Case 3: p_val extends a sequence: N1 - N2 - p_val
                    # 來源：新大腦.pdf - EXT_GM7 Case 3 (Page 31)
                    r_n1_ext2, c_n1_ext2 = r_idx - 2 * dr, c_idx - 2 * dc #
                    r_n2_ext2, c_n2_ext2 = r_idx - dr, c_idx - dc
                    
                    if 0 <= r_n1_ext2 < rows and 0 <= c_n1_ext2 < cols and \
                       0 <= r_n2_ext2 < rows and 0 <= c_n2_ext2 < cols: #
                        val_n1_ext2 = grid[r_n1_ext2, c_n1_ext2]
                        val_n2_ext2 = grid[r_n2_ext2, c_n2_ext2]

                        if val_n1_ext2 != -1 and val_n2_ext2 != -1: #
                            # Check for N2=N1+d, p_val=N1+2d => val_n2_ext2 - val_n1_ext2 == p_val - val_n2_ext2 (d)
                            # 來源：新大腦.pdf - EXT_GM7 Case 3 logic (Page 31-32) #
                            # PDF: `common_diff = val_n2_ext2 - val_n1_ext2`
                            # `if common_diff !=0 and p_val == val_n2_ext2 + common_diff:`
                            common_diff = val_n2_ext2 - val_n1_ext2 #
                            if not math.isclose(common_diff,0) and math.isclose(p_val, val_n2_ext2 + common_diff): # 來源：新大腦.pdf (Page 32) - Corrected index typo c_idx-1 to c_idx-dc for general direction
                                max_cell_gap_score = max(max_cell_gap_score, config.score_arithmetic_generic_extend)
            
            # PDF had current_pval_score > max_cell_gap_score, but current_pval_score wasn't updated per direction.
            # max_cell_gap_score is directly updated.
            # 來源：新大腦.pdf - EXT_GM7 Normalization (Page 32)
            scores[r_idx, c_idx] = MathUtils.normalize_value(max_cell_gap_score, 0, 1.0, clamp=True) # Scores from map are ~0-1 #
    
    return scores * config.weight


# 來源：新大腦.pdf - 14. EXT_GM8_Edge_Affinity_Vec (Page 31)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM8強化建議
# Config for this (EdgeAffinityConfig) was defined in PART 2
def EXT_GM8_Edge_Affinity_Vec(
    grid: np.ndarray,
    config: EdgeAffinityConfig,
    request_id: str | None = "N/A_GM8_EdgeAff",
) -> np.ndarray:
    """
    (GM8-邊緣親和度)
    核心規則:評估空格與盤面邊緣或角落的接近程度及其策略意義。
    目的:根據策略配置,偏好靠近或遠離邊緣/角落的空格。 #
    啟發式類型:位置與邊界分析
    輸出詮釋:分數高低取決於設定(偏好/避開邊緣)。預設偏好邊緣,越靠近邊緣/角落分數越高。
    來源：新大腦.pdf - EXT_GM8_Edge_Affinity_Vec (Page 31)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM8"
    logger.debug(
        f"Executing EXT_GM8_Edge_Affinity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 32) #
        return scores

    affinity_mode = config.affinity_mode # 來源：新大腦.pdf (Page 32)
    corner_bonus_prefer = config.corner_bonus_prefer # 來源：新大腦.pdf (Page 32)
    corner_penalty_avoid = config.corner_penalty_avoid # 來源：新大腦.pdf (Page 32)

    # 來源：新大腦.pdf - EXT_GM8 Max possible minimum distance to an edge (Page 32)
    # This would be for a cell at the center of the board.
    max_min_dist_to_edge_row = (rows - 1) // 2 if rows > 0 else 0 #
    max_min_dist_to_edge_col = (cols - 1) // 2 if cols > 0 else 0
    
    # The actual maximum of minimum distances to any edge for any cell on the board.
    # For a cell at (r,c), its min_dist_to_edge is min(r, rows-1-r, c, cols-1-c).
    # The max value this min_dist_to_edge can take is at the center.
    # PDF calculation: float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col))
    # This seems correct: e.g. 5x7 grid, center is (2,3). min_dist_row=2, min_dist_col=3.
    # max_min_dist_row=(5-1)//2 = 2. max_min_dist_col=(7-1)//2 = 3. min(2,3)=2. Correct.
    overall_max_of_min_distances = float(min(max_min_dist_to_edge_row, max_min_dist_to_edge_col)) # 來源：新大腦.pdf (Page 33) #
    
    # 來源：新大腦.pdf - EXT_GM8 Handle overall_max_of_min_distances == 0 (Page 33)
    # If overall_max_of_min_distances is 0 (e.g., a 1xN or 2xN line, or 1x1, 2x1, 2x2),
    # it means all cells are on an edge or one step from it.
    if math.isclose(overall_max_of_min_distances, 0.0) and (rows > 0 and cols > 0): # For non-empty grid #
        if rows <= 2 or cols <= 2 : # For very thin/small grids where center is edge/near-edge
             overall_max_of_min_distances = 0.5 # Avoid div by zero, gives some scale for normalization
                                    
            # All cells on such grids will have min_dist 0 or 1.
                                               # If min_dist is 0, normalized_dist will be 0.
                                               # If min_dist is 1, normalized_dist will be 1/0.5=2 (needs clamp).
        else: # This case should not be hit if logic for max_min_dist_... is correct #
             overall_max_of_min_distances = 1.0 # Fallback if it's calculated as 0 for larger grids.
    if overall_max_of_min_distances <= 0 : overall_max_of_min_distances = 1.0 # General fallback to prevent div by zero #


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 33)
                continue

            dist_to_top_edge = float(r_idx)
            dist_to_bottom_edge = float(rows - 1 - r_idx) #
            dist_to_left_edge = float(c_idx)
            dist_to_right_edge = float(cols - 1 - c_idx)

            min_dist = min(dist_to_top_edge, dist_to_bottom_edge, dist_to_left_edge, dist_to_right_edge) # 來源：新大腦.pdf (Page 33)

            is_corner = (r_idx == 0 or r_idx == rows - 1) and \
                        (c_idx == 0 or c_idx == cols - 1) # 來源：新大腦.pdf (Page 33) #
            
            current_score: float = 0.0
            normalized_dist: float = 0.0

            if overall_max_of_min_distances > 1e-6: # Use tolerance for float comparison
                normalized_dist = min_dist / overall_max_of_min_distances #
                normalized_dist = min(1.0, max(0.0, normalized_dist)) # Clamp # 來源：新大腦.pdf (Page 33)
            elif math.isclose(min_dist, 0.0): # All cells are on an edge, min_dist is 0
                normalized_dist = 0.0 # 來源：新大腦.pdf (Page 33)
            else: # Should not happen if overall_max_of_min_distances is handled
                normalized_dist = 1.0 # 來源：新大腦.pdf (Page 33) #

            if affinity_mode == "prefer_edge": # 來源：新大腦.pdf (Page 33)
                current_score = 1.0 - normalized_dist  # Closer to edge (smaller dist) -> higher score
                if is_corner and math.isclose(min_dist, 0.0): # Only apply corner bonus if truly on edge
                    current_score += corner_bonus_prefer #
            elif affinity_mode == "avoid_edge": # 來源：新大腦.pdf (Page 33)
                current_score = normalized_dist  # Further from edge (larger dist) -> higher score
                if is_corner and math.isclose(min_dist, 0.0):
                    current_score -= corner_penalty_avoid #
            
            # Normalize final score to be between 0 and 1, considering bonus/penalty
            # Max possible score: 1.0 + corner_bonus_prefer
            # Min possible score: 0.0 - corner_penalty_avoid
            # 來源：新大腦.pdf - EXT_GM8 Final Normalization (Page 34)
            # The PDF normalizes using (-corner_penalty_avoid, 1.0 + corner_bonus_prefer)
            # This seems problematic if penalty makes it <0 and bonus >1, then normalize to 0-1.
            # It's better to clamp the current_score first, then normalize if needed,
            # or ensure the normalization range is correct.
            # The scores are already conceptually 0-1 before bonus/penalty.
            # Let's clamp current_score to a reasonable range like [0, 1 + corner_bonus_prefer]
            # and then normalize that range to [0,1].
            # Simpler: clamp result to [0,1] after bonus/penalty.
            current_score = max(0.0, min(1.0 + corner_bonus_prefer, current_score)) # Clamp to possible range
            # Now, normalize this to [0,1] if the range is not already [0,1]
            # If prefer_edge, range is [0, 1+bonus].
            # If avoid_edge, range is [0-penalty, 1].
            # The MathUtils.normalize_value in PDF has range [-CP_avoid, 1+CB_prefer] which implies
            # the value `current_score` can be negative.
            # Let's use the PDF's normalization directly:
            min_norm_range = 0.0 - corner_penalty_avoid if affinity_mode == "avoid_edge" else 0.0 #
            max_norm_range = 1.0 + corner_bonus_prefer if affinity_mode == "prefer_edge" else 1.0
            if math.isclose(max_norm_range, min_norm_range) : # if bonus and penalty are such that range is zero #
                 max_norm_range = min_norm_range + 1.0 # ensure non-zero range for normalization


            scores[r_idx, c_idx] = MathUtils.normalize_value(current_score,
                                                            min_val=min_norm_range, 
                                                            max_val=max_norm_range,  #
                                                            clamp=True) #
            # Final clamp just in case, though normalize_value with clamp=True should handle it.
            scores[r_idx, c_idx] = max(0.0, min(1.0, scores[r_idx, c_idx])) #


    return scores * config.weight

# 來源：新大腦.pdf - 15. EXT_GM9_Center_Control_Vec (Page 34)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM9強化建議
# Config for this (CenterControlConfig) was defined in PART 2
def EXT_GM9_Center_Control_Vec(
    grid: np.ndarray,
    config: CenterControlConfig,
    request_id: str | None = "N/A_GM9_CenterCtrl",
) -> np.ndarray:
    """
    (GM9-中心控制偏好)
    核心規則:評估空格與盤面中心的接近程度及其策略意義。
    目的:根據策略配置,偏好靠近或遠離盤面中心區域的空格。 #
    啟發式類型:位置與中心性分析
    輸出詮釋:分數高低取決於設定(偏好/避開中心)。預設偏好中心,越靠近中心分數越高。
    來源：新大腦.pdf - EXT_GM9_Center_Control_Vec (Page 34)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM9"
    logger.debug(
        f"Executing EXT_GM9_Center_Control_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 34) #
        return scores

    affinity_mode = config.affinity_mode # 來源：新大腦.pdf (Page 34)
    
    center_r = (rows - 1) / 2.0 # 來源：新大腦.pdf (Page 34)
    center_c = (cols - 1) / 2.0 # 來源：新大腦.pdf (Page 34)

    # Max possible distance from any cell to the center is the distance from a corner to the center.
    # 來源：新大腦.pdf - EXT_GM9 max_dist_to_center (Page 34) #
    # Using (0,0) as the reference corner.
    max_dist_to_center = MathUtils.euclidean_distance((0.0, 0.0), (center_r, center_c)) #

    # 來源：新大腦.pdf - EXT_GM9 Handle max_dist_to_center == 0 (Page 34)
    if math.isclose(max_dist_to_center, 0.0) : # if grid is 1x1 or effectively so #
        if rows <= 1 and cols <= 1: # Truly a 1x1 or 0x0 grid (0x0 caught by early return)
            # For a 1x1 grid, all cells are the center.
            # Score should be neutral or max depending on interpretation.
            # If prefer_center, score should be high (1.0).
            # If avoid_center, low (0.0).
            # The normalization logic MathUtils.normalize_value(0,0,0) returns 0.5.
            # if affinity_mode == "prefer_center": scores[0,0] = 1.0 * config.weight (if 1x1)
            # else: scores[0,0] = 0.0 * config.weight
            # This is handled by the loop, normalized_dist will be 0.5 from normalize_value if max_dist_to_center is 0.
            # current_score will then be 0.5 or 0.5.
            # Let's refine the max_dist_to_center for 1x1.
            pass # max_dist_to_center remains 0, MathUtils.normalize_value will give 0.5 for dist=0
        else: # Calculated as 0 for larger grids (should not happen if center_r/c are correct for >1x1)
            max_dist_to_center = 1.0 # Fallback to prevent div by zero if logic error

    # Ensure max_dist_to_center is not zero if grid is larger than 1x1 to avoid division by zero
    # or to give meaningful normalization.
    if math.isclose(max_dist_to_center, 0.0) and (rows > 1 or cols > 1): #
         max_dist_to_center = 1.0 # Should not be hit if center_r, center_c are calculated for >1x1

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 35)
                continue

            current_dist_to_center = MathUtils.euclidean_distance( #
                (float(r_idx), float(c_idx)), (center_r, center_c)
            ) # 來源：新大腦.pdf (Page 35)

            normalized_dist: float
            if max_dist_to_center > 1e-6: # Use tolerance
                normalized_dist = MathUtils.normalize_value(
                    current_dist_to_center, 0, max_dist_to_center, clamp=True #
                ) # 來源：新大腦.pdf (Page 35)
            elif math.isclose(current_dist_to_center, 0.0): # For 1x1 grid, dist is 0, max_dist is 0.
                normalized_dist = 0.0 # Perfectly at center means 0 distance.
                # MathUtils.normalize_value(0,0,0) = 0.5.
                                     # If we want 0 dist to result in max score for "prefer_center" (1.0 - 0.0 = 1.0),
                                     # then normalized_dist = 0 is correct.
                # 來源：新大腦.pdf - EXT_GM9 Discussion on 1x1 grid norm (Page 35) #
            else: # Should not be reached if max_dist_to_center handled correctly
                normalized_dist = 1.0


            current_score: float
            if affinity_mode == "prefer_center": # 來源：新大腦.pdf (Page 35)
                current_score = 1.0 - normalized_dist  # Closer to center (smaller dist) -> higher score #
            elif affinity_mode == "avoid_center": # 來源：新大腦.pdf (Page 35)
                current_score = normalized_dist  # Further from center (larger dist) -> higher score
            else: # Should not happen with Pydantic validation
                current_score = 0.5 
            
            # Final score is already in [0,1] due to normalized_dist being [0,1]
            # PDF uses MathUtils.normalize_value(current_score, 0, 1.0, clamp=True) which is fine.
            # 來源：新大腦.pdf - EXT_GM9 Final clamp (Page 35) #
            scores[r_idx, c_idx] = MathUtils.normalize_value(current_score, 0, 1.0, clamp=True)
            
    return scores * config.weight

# 來源：新大腦.pdf - 16. EXT_GM10_Blocking_Value_Vec (Page 35)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM10強化建議
# Config for this (BlockingValueConfig) was defined using the updated version
def EXT_GM10_Blocking_Value_Vec(
    grid: np.ndarray,
    config: BlockingValueConfig,
    request_id: str | None = "N/A_GM10_Blocking",
) -> np.ndarray:
    """
    (GM10-阻斷價值評估)
    核心規則:評估在空格填入數字是否能有效「阻止」或「避免」形成預定義的不良模式或序列。
    目的:偏好那些不會導致形成不良結構的填補,或者理想情況下能主動阻止潛在不良結構形成的填補。
    啟發式類型:防禦性策略與模式規避
    輸出詮釋:分數越高表示在該空格填入數字後,越不可能形成已知的不良模式。
    來源：新大腦.pdf - EXT_GM10_Blocking_Value_Vec (Page 35) #
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM10"
    logger.debug(
        f"Executing EXT_GM10_Blocking_Value_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 36)
        return scores #

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 36)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 36)
        return scores

    UNDESIRABLE_SEQUENCES = [seq for seq in config.undesirable_sequences_list if len(seq) == config.check_line_length]
    line_len_to_check = config.check_line_length

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 36)
                continue #
            
            # Score for this cell will be the max score (safety) achievable by placing any potential number
            # 來源：新大腦.pdf (Page 36)
            max_safety_score_for_cell: float = 0.0 # Default to low if all placements are bad (or if no potential numbers)
            
            # PDF logic: "if not potential_numbers_to_place: scores[r_idx,c_idx]=0.5" is not quite right here,
            # as we already checked potential_numbers_to_place at the function start.
            # If loop doesn't run, max_safety_score_for_cell remains 0.0.
            
            at_least_one_pval_evaluated = False
            for p_val in potential_numbers_to_place:
                at_least_one_pval_evaluated = True
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                forms_undesirable_pattern_for_pval = False # Renamed for clarity #

                # Check lines of 'line_len_to_check' passing through (r_idx, c_idx)
                # Directions: Horizontal, Vertical, Main Diagonal, Anti-Diagonal
                # 來源：新大腦.pdf - EXT_GM10 Directions (Page 36)
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if forms_undesirable_pattern_for_pval: break # Already found one for this p_val #

                    # Iterate through all windows of 'line_len_to_check' that include (r_idx, c_idx)
                    # where (r_idx, c_idx) is now filled with p_val.
                    # 來源：新大腦.pdf - EXT_GM10 Offset logic (Page 37) #
                    # offset is the starting position of the window relative to p_val's position in the window
                    for i_offset_in_window in range(line_len_to_check):
                        current_line_values_list: List[int] = []
                        is_valid_segment = True #
                        
                        for k_in_segment in range(line_len_to_check):
                            # Position of k_in_segment-th cell in the current line window #
                            eval_r = r_idx + (k_in_segment - i_offset_in_window) * dr_line
                            eval_c = c_idx + (k_in_segment - i_offset_in_window) * dc_line

                            if not (0 <= eval_r < rows and 0 <= eval_c < cols): #
                                is_valid_segment = False
                                break
                            current_line_values_list.append(int(temp_grid[eval_r, eval_c])) #
                        
                        if is_valid_segment: # No need to check len, it's always line_len_to_check
                            # PDF: "Ensure the currently placed p_val at (r_idx,c_idx) is part of this line" #
                            # This is implicitly true by how the window is constructed around (r_idx,c_idx).
                            # 來源：新大腦.pdf (Page 37)

                            for undesirable_seq in UNDESIRABLE_SEQUENCES:
                                # PDF: current_line_values == undesirable_seq #
                                # Ensure types are consistent if undesirable_seq stores ints #
                                if current_line_values_list == undesirable_seq:
                                    forms_undesirable_pattern_for_pval = True
                                    break # Found an undesirable pattern for this line #
                            if forms_undesirable_pattern_for_pval: break # For this direction
                    if forms_undesirable_pattern_for_pval: break # For this p_val
                
                current_score_for_pval = config.score_if_safe if not forms_undesirable_pattern_for_pval else config.score_if_unsafe #
                # 來源：新大腦.pdf (Page 37) - PDF has 0.9 if not, 0.1 if yes.
                if current_score_for_pval > max_safety_score_for_cell: #
                    max_safety_score_for_cell = current_score_for_pval
            
            if not at_least_one_pval_evaluated and not potential_numbers_to_place : # Should have been caught earlier
                 scores[r_idx,c_idx] = 0.5 # Neutral if no options and somehow reached here
            else: #
                 scores[r_idx, c_idx] = max_safety_score_for_cell # 來源：新大腦.pdf (Page 37-38) - Corrected var name

    return scores * config.weight

# 來源：新大腦.pdf - 17. EXT_GM11_Pair_Correlation_Vec (Page 38)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM11強化建議
# Config for this (PairCorrelationConfig) was defined in PART 2
def EXT_GM11_Pair_Correlation_Vec(
    grid: np.ndarray,
    config: PairCorrelationConfig,
    request_id: str | None = "N/A_GM11_PairCorr", #
) -> np.ndarray:
    """
    (GM11-數字配對關聯分析)
    核心規則:分析特定數字對(pair)共同出現或以特定相對位置(此處為鄰近)出現的頻率與價值。
    目的:偏好那些能夠形成已知有利數字配對的填補。
    啟發式類型: 關聯性分析(局部)
    輸出詮釋:分數越高表示在該空格填入特定數字後,能與周圍已存在的數字形成更多或更高價值的有利配對。
    來源：新大腦.pdf - EXT_GM11_Pair_Correlation_Vec (Page 38)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM11"
    logger.debug(
        f"Executing EXT_GM11_Pair_Correlation_Vec with config: {config.model_dump_json(indent=2)}", # Pydantic config
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape #
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 38)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 38)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 38)
        return scores

    # 來源：新大腦.pdf - EXT_GM11 FAVORABLE_PAIRS_SCORES (Page 38)
    # Using config.favorable_pairs
    FAVORABLE_PAIRS_SCORES = {tuple(sorted(k)): v for k,v in config.favorable_pairs.items()} # Normalize key order for easier lookup if desired, though PDF implies (p_val, neighbor_val) order #

    max_single_pair_score: float = 0.0 # 來源：新大腦.pdf (Page 38)
    if FAVORABLE_PAIRS_SCORES: # Check if FAVORABLE_PAIRS_SCORES from config resulted in anything
      if config.favorable_pairs: # Check original config source
        max_single_pair_score = float(max(config.favorable_pairs.values()))
    
    # Heuristic max possible score: if all 8 neighbors form max-scoring pairs
    # 來源：新大腦.pdf - EXT_GM11 heuristic_max_total_pair_score (Page 39)
    heuristic_max_total_pair_score = 8.0 * max_single_pair_score if max_single_pair_score > 1e-6 else 1.0

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 39) #
                continue

            max_accumulated_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 39)

            for p_val in potential_numbers_to_place:
                current_pval_accumulated_score: float = 0.0
                
                # Check 8 neighbors #
                # 來源：新大腦.pdf - EXT_GM11 Check 8 neighbors (Page 39)
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: #
                            continue

                        nr, nc = r_idx + dr, c_idx + dc

                        if 0 <= nr < rows and 0 <= nc < cols: #
                            neighbor_val = grid[nr, nc]
                            if neighbor_val != -1:  # If neighbor is an existing number
                                # Check if (p_val, neighbor_val) is a favorable pair #
                                # 來源：新大腦.pdf - EXT_GM11 Check favorable pair (Page 39)
                                # The PDF has: if (p_val, int(neighbor_val)) in FAVORABLE_PAIRS_SCORES: #
                                # This implies the order matters, or keys in FAVORABLE_PAIRS_SCORES should handle both orders or be normalized.
                                # Using the direct tuple (p_val, int(neighbor_val)) as key. #
                                pair_key = (p_val, int(neighbor_val))
                                if pair_key in config.favorable_pairs: # Use config directly
                                    current_pval_accumulated_score += config.favorable_pairs[pair_key]
                                # PDF original: current_pval_accumulated_score += FAVORABLE_PAIRS_SCORES[(p_val, int(neighbor_val))]
                                    # The PDF also had `current_pval_accumulated_score += 1` which seems like a typo if scores are provided.
                                # I am using the score from the map. #

                if current_pval_accumulated_score > max_accumulated_score_for_cell: # 來源：新大腦.pdf (Page 39)
                    max_accumulated_score_for_cell = current_pval_accumulated_score
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_accumulated_score_for_cell, 0, heuristic_max_total_pair_score, clamp=True
            ) # 來源：新大腦.pdf (Page 39) #
            
    return scores * config.weight


# 來源：新大腦.pdf - 18. EXT_GM12_Island_Analysis_Vec (Page 39)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM12強化建議
# Config for this (IslandAnalysisConfig) was defined in PART 2
def EXT_GM12_Island_Analysis_Vec(
    grid: np.ndarray,
    config: IslandAnalysisConfig,
    request_id: str | None = "N/A_GM12_Island",
) -> np.ndarray:
    """
    (GM12 - 島嶼分析)
    核心規則:分析由已填數字形成的「島嶼」的特性,如大小、緊湊度和平均值。
    目的:根據策略,可能偏好大型、緊湊或包含高價值數字的島嶼。此處假設偏好較大、較緊湊、平均值較高的數字島嶼。
    啟發式類型:連通元件與區域形態分析(針對已填數字)
    輸出詮释: 分數越高表示該格屬於一個更優(大、緊湊、高平均值)的數字島嶼。空格得0分。
    來源：新大腦.pdf - EXT_GM12_Island_Analysis_Vec (Page 39-40)
    """
    if not config.enabled: #
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM12"
    logger.debug(
        f"Executing EXT_GM12_Island_Analysis_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) # Empty cells get 0 score from this module
    # 來源：新大腦.pdf (Page 40, 41)
    if rows == 0 or cols == 0:
        return scores #

    visited_island_search = np.zeros_like(grid, dtype=bool) # 來源：新大腦.pdf (Page 40)
    max_val_on_board = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 40)
    if max_val_on_board == 0: max_val_on_board = 1.0 # Avoid div by zero # 來源：新大腦.pdf (Page 40)

    # Weights from config
    w_size = config.w_size # 來源：新大腦.pdf (Page 40)
    w_compactness = config.w_compactness # 來源：新大腦.pdf (Page 40)
    w_avg_value = config.w_avg_value # 來源：新大腦.pdf (Page 40)

    for r_start in range(rows):
        for c_start in range(cols): #
            # Found an unvisited *number* (island part)
            if grid[r_start, c_start] != -1 and not visited_island_search[r_start, c_start]: # 來源：新大腦.pdf (Page 40)
                current_island_cells: List[Tuple[int, int]] = [] # 來源：新大腦.pdf (Page 40)
                current_island_values: List[int] = [] # 來源：新大腦.pdf (Page 40)
                
                q = deque([(r_start, c_start)]) #
                visited_island_search[r_start, c_start] = True
                
                min_r_bbox, max_r_bbox = r_start, r_start # 來源：新大腦.pdf (Page 40)
                min_c_bbox, max_c_bbox = c_start, c_start # 來源：新大腦.pdf (Page 40) #

                while q: # 來源：新大腦.pdf (Page 40)
                    r_curr, c_curr = q.popleft()
                    current_island_cells.append((r_curr, c_curr))
                    current_island_values.append(int(grid[r_curr, c_curr]))

                    min_r_bbox = min(min_r_bbox, r_curr) # 來源：新大腦.pdf (Page 40-41) #
                    max_r_bbox = max(max_r_bbox, r_curr)
                    min_c_bbox = min(min_c_bbox, c_curr)
                    max_c_bbox = max(max_c_bbox, c_curr)

                    # 4-connectivity for islands #
                    # 來源：新大腦.pdf - EXT_GM12 4-connectivity (Page 41)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: 
                        nr, nc = r_curr + dr, c_curr + dc #
                        if 0 <= nr < rows and 0 <= nc < cols and \
                           grid[nr, nc] != -1 and not visited_island_search[nr, nc]: # 來源：新大腦.pdf (Page 41)
                            visited_island_search[nr, nc] = True #
                            q.append((nr, nc))
                
                # Calculate island characteristics
                # 來源：新大腦.pdf - EXT_GM12 Island characteristics (Page 41)
                island_size = float(len(current_island_cells)) #
                avg_value_island: float = 0.0
                if island_size > 0:
                    avg_value_island = sum(current_island_values) / island_size
                
                bbox_height = float(max_r_bbox - min_r_bbox + 1) #
                bbox_width = float(max_c_bbox - min_c_bbox + 1)
                bbox_area = bbox_height * bbox_width
                
                compactness: float = 0.0
                if bbox_area > 0: # Avoid division by zero #
                    compactness = island_size / bbox_area # (Ratio of actual cells to bounding box area)

                # Normalize characteristics
                # 來源：新大腦.pdf - EXT_GM12 Normalize characteristics (Page 41)
                norm_size = MathUtils.normalize_value(island_size, 1, float(rows * cols), clamp=True) #
                norm_compactness = MathUtils.normalize_value(compactness, 0, 1.0, clamp=True) # Already 0-1
                norm_avg_value = MathUtils.normalize_value(avg_value_island, 1, max_val_on_board, clamp=True)

                # Combine into a single island score
                # 來源：新大腦.pdf - EXT_GM12 Combine island score (Page 41) #
                island_score_unnormalized = (
                    w_size * norm_size +
                    w_compactness * norm_compactness +
                    w_avg_value * norm_avg_value
                ) #
                # Normalize combined score (max possible is sum of weights if they sum to 1)
                total_weights = w_size + w_compactness + w_avg_value
                max_possible_island_score = total_weights if total_weights > 0 else 1.0

                final_island_score = MathUtils.normalize_value(island_score_unnormalized, 0, max_possible_island_score, clamp=True) #
                # PDF: MathUtils.normalize_value(island_score, 0, 1.0, clamp=True) - assumes weights sum to 1 or less.
                # Assign this score to all cells in the current island #
                # 來源：新大腦.pdf - EXT_GM12 Assign score (Page 41)
                for r_cell, c_cell in current_island_cells:
                    scores[r_cell, c_cell] = final_island_score
            
            
            elif grid[r_start, c_start] == -1: # Empty cells get 0 score (already initialized) # 來源：新大腦.pdf (Page 41) #
                # Ensure visited_overall is marked for empty cells too to avoid re-processing them as start points
                # for an "island search" that would immediately terminate.
                # The logic `grid[r_start,c_start]!=-1 and not visited_island_search` handles this. #
                pass # Scores remain 0 for empty cells
            
            # Mark as visited to avoid re-check even if it's an empty cell we skipped.
            # The primary visited_island_search is for actual island cells.
            # No, only mark actual island cells or cells part of a processed component.
            # Empty cells are handled by the first `if` in the loop.
    return scores * config.weight #