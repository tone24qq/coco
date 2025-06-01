# brain3.py
# Part 3 of 3: Contains the final set of AI scoring modules,
# module registration, dispatch logic, and the main verification block.
# Based on Brain.txt, which was generated according to 新大腦.pdf, 给你2025资料在深度建议一次.pdf, 极限强化.pdf

import numpy as np
import math
from collections import Counter, deque # deque might not be used by these specific modules, Counter may be
import logging
from typing import List, Dict, Tuple, Callable, Optional, Any, Set

from pydantic import BaseModel, Field

# Assuming brain1.py is in the same path and contains these definitions
from brain1 import BaseModuleConfig, MathUtils, BoardAnalyzerUtils

# Imports for module functions and configs from brain1 and brain2 for registration
# These will be used to populate REGISTERED_MODULES_BRAIN and DEFAULT_MODULE_CONFIGS
from brain1 import (
    WeightedProximityConfig, LocalHeterogeneityConfig, PotentialFieldConfig,
    DiscontinuityRepairConfig, PathfindingValueConfig, ResourceControlConfig,
    LineControlConfig, ConnectedComponentConfig,
    EXT_A2_Weighted_Proximity_Vec, EXT_M3_Local_Heterogeneity_Vec,
    EXT_D3_Potential_Field_Vec, EXT_F10_Discontinuity_Vec,
    EXT_P7_Pathfinding_Value_Vec, EXT_R5_Resource_Control_Vec,
    EXT_GM1_Row_Control_Vec, EXT_GM2_Col_Flow_Vec, EXT_GM3_Adv_Connected_Comp_Vec
)
from brain2 import (
    SpatialAutocorrelationConfig, LineCompletionConfig, SymmetryPotentialConfig,
    NumericGapsConfig, EdgeAffinityConfig, CenterControlConfig,
    BlockingValueConfig as BlockingValueConfigBrain2, # Alias to avoid potential name clash if defined locally
    PairCorrelationConfig, IslandAnalysisConfig,
    EXT_GM4_Spatial_Auto_Corr_Vec, EXT_GM5_Line_Completion_Vec,
    EXT_GM6_Symmetry_Potential_Vec, EXT_GM7_Numeric_Gaps_Vec,
    EXT_GM8_Edge_Affinity_Vec, EXT_GM9_Center_Control_Vec,
    EXT_GM10_Blocking_Value_Vec, EXT_GM11_Pair_Correlation_Vec,
    EXT_GM12_Island_Analysis_Vec
)


logger = logging.getLogger(__name__)

# --- Pydantic Config Models for Modules (Modules 19-26) ---

class SequenceDiversityConfig(BaseModuleConfig): # For GM13
    # 來源：新大腦.pdf - EXT_GM13 parameters (Page 42) #
    short_sequence_len: int = Field(default=3, ge=2) #
    # Heuristic max_distinct_sequences used for normalization, not directly a config for behavior
    # Could add weights for different types of diverse sequences (arithmetic vs identical)


class RiskAssessmentConfig(BaseModuleConfig): # For GM14
    # 來源：新大腦.pdf - EXT_GM14 parameters (Page 44)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM14 靈活性度量的複雜化 #
    flexibility_metric_mode: str = Field(default="subsequent_moves", pattern="^(subsequent_moves|product_moves_empty_cells)$")


class InformationGainConfig(BaseModuleConfig): # For GM15
    # 來源：新大腦.pdf - EXT_GM15 parameters (Page 45-46)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM15 熵計算的對象 #
    entropy_scope: str = Field(default="global_full", pattern="^(global_full|global_filled_only)$", description="熵計算範圍：global_full (含-1), global_filled_only (不含-1)")


class HarmonicCentralityConfig(BaseModuleConfig): # For GM16
    # 來源：新大腦.pdf - EXT_GM16 parameters (Page 47)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM16 節點的定義 #
    node_definition: str = Field(default="all_cells", pattern="^(all_cells|empty_cells_only|filled_cells_only)$", description="計算調和中心性時考慮的節點類型")


class LocalEntropyMinimizationConfig(BaseModuleConfig): # For GM17
    # 來源：新大腦.pdf - EXT_GM17 parameters (Page 48) #
    radius: int = Field(default=1, ge=1, description="局部鄰域半徑")
    # max_local_entropy_change is for normalization, calculated internally


class RLValueEstimationConfig(BaseModuleConfig): # For GM18
    # 來源：新大腦.pdf - EXT_GM18 parameters (Page 50-51)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM18 特徵庫的擴展與優化
    # Feature weights would ideally be loaded or learned. #
    feature_weights: Dict[str, float] = Field(default_factory=lambda: { #
        "identical_3": 1.0,
        "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.1, # 來源：新大腦.pdf (Page 51)
        "edge_affinity_boost": 0.05,   # 來源：新大腦.pdf (Page 52)
    })
    # More features could be added here with their weights


class SkipPatternConfig(BaseModuleConfig): # For GM19
    # 來源：新大腦.pdf - EXT_GM19 parameters (Page 53-54)
    min_occurrences_for_pattern_factor: float = Field(default=0.05, ge=0.0, le=1.0, description="形成主導跳格模式所需的最少出現次數（佔總跳格數的比例）") # PDF uses 0.05 of len(skip_vector_tuples_list) #
    base_pattern_definition: str = Field(default="left_to_right_top_to_bottom", description="理論基礎位置的掃描模式（概念性）")


class SkipPatternConfidenceConfig(BaseModuleConfig): # For GM20
    # 來源：新大腦.pdf - EXT_GM20 parameters (Page 55-56)
    min_occurrences_for_pattern_factor_gm20: float = Field(default=0.05, ge=0.0, le=1.0) # Same as GM19's factor
    # 來源：新大腦.pdf - EXT_GM20 arithmetic sequence enhancement (Page 57)
    arithmetic_enhancement_bonus: float = Field(default=0.4, ge=0.0, description="形成一致等差序列的增強因子")
    internal_gap_fill_bonus: float = Field(default=0.1, ge=0.0, description="填充內部間隙形成等差序列的額外獎勵")


# --- Scoring Module Implementations (Modules 19-26) ---

# 來源：新大腦.pdf - 19. EXT_GM13_Sequence_Diversity_Vec (Page 41)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM13強化建議
# Config for this (SequenceDiversityConfig) was defined in PART 2
def EXT_GM13_Sequence_Diversity_Vec( #
    grid: np.ndarray,
    config: SequenceDiversityConfig,
    request_id: str | None = "N/A_GM13_SeqDiv",
) -> np.ndarray:
    """
    (GM13-序列多樣性)
    核心規則:評估填補位置是否有助於形成多樣化的短序列(例如,不同方向、不同類型),而非僅專注於單一長序列。
    目的:鼓勵在盤面上形成多個不同類型或方向的短數字序列,增加盤面的「活性」或「機會」。
    啟發式類型:模式識別與組合多樣性
    輸出詮釋:分數越高表示在該空格填入特定數字後,能參與形成的獨特短序列種類越多。
    來源：新大腦.pdf - EXT_GM13_Sequence_Diversity_Vec (Page 41-42) #
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM13"
    logger.debug(
        f"Executing EXT_GM13_Sequence_Diversity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 42)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 42) #
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 42)
        return scores

    short_sequence_len = config.short_sequence_len # 來源：新大腦.pdf (Page 42)
    # Max distinct short sequences a single cell might participate in (heuristic for normalization)
    # 來源：新大腦.pdf - EXT_GM13 heuristic_max_distinct_sequences (Page 42)
    # For length 3, in 4 directions, cell can be in 3 positions.
    # Max 4*2 types (arith, ident) = 8. #
    # This is a rough upper bound.
    heuristic_max_distinct_sequences = 8.0  #
    if short_sequence_len != 3: # Adjust if length changes
        heuristic_max_distinct_sequences = float(4 * 2 * (short_sequence_len)) # Very rough

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 42)
                continue

            max_diversity_count_for_cell: int = 0 # 來源：新大腦.pdf (Page 42) #

            for p_val in potential_numbers_to_place:
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                
                # Store signatures like ("arithmetic", (dr,dc), diff) or ("identical", (dr,dc), val) #
                # 來源：新大腦.pdf - EXT_GM13 found_sequence_signatures (Page 42)
                found_sequence_signatures: Set[Tuple[str, Tuple[int,int], int]] = set() 

                # Check in 4 directions (H, V, D1, D2)
                # 來源：新大腦.pdf - EXT_GM13 Directions (Page 42)
                for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]: #
                    # For each direction, check 'short_sequence_len' possible alignments of a sequence
                    # where p_val (at (r_idx, c_idx)) is involved.
                    # i_offset_in_window: position of p_val within the current window of 'short_sequence_len' #
                    # 來源：新大腦.pdf - EXT_GM13 i_offset loop (Page 42)
                    for i_offset_in_window in range(short_sequence_len):
                        current_sequence_values: List[int] = []
                        valid_segment = True #
                        
                        for k_in_segment in range(short_sequence_len):
                            # Position of k_in_segment-th element in the window, relative to (r_idx, c_idx) #
                            # 來源：新大腦.pdf - EXT_GM13 check_r, check_c calculation (Page 43)
                            eval_r = r_idx + (k_in_segment - i_offset_in_window) * dr_dir
                            eval_c = c_idx + (k_in_segment - i_offset_in_window) * dc_dir #

                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                valid_segment = False
                                break #
                            current_sequence_values.append(int(temp_grid[eval_r, eval_c]))
                        
                        if valid_segment: # Implicitly len(current_sequence_values) == short_sequence_len #
                            # Analyze this short sequence (s)
                            s = current_sequence_values
                            # All values must be non -1 (which is true since temp_grid has p_val and others are from original or p_val) #
                            
                            # 1. Arithmetic sequence (non-constant)
                            # 來源：新大腦.pdf - EXT_GM13 Arithmetic check (Page 43) #
                            if len(s) >= 2 : # Need at least 2 to check diff
                                diffs = [s[k+1] - s[k] for k in range(len(s)-1)]
                                if diffs: # Ensure diffs is not empty #
                                    first_diff = diffs[0]
                                    if all(math.isclose(d, first_diff) for d in diffs) and not math.isclose(first_diff, 0): #
                                        # Normalize direction vector for signature uniqueness (e.g., (0,1) is same as (0,-1) for line orientation)
                                        norm_dr = abs(dr_dir) if dc_dir == 0 else dr_dir # Simple normalization #
                                        norm_dc = abs(dc_dir) if dr_dir == 0 else dc_dir
                                        if norm_dr == 1 and norm_dc == 1 and norm_dr * norm_dc < 0: # anti-diag normalize (1,-1) vs (-1,1) #
                                            norm_dr, norm_dc = min(abs(dr_dir),dr_dir), min(abs(dc_dir),dc_dir) if dr_dir != dc_dir else dc_dir

                                        found_sequence_signatures.add(("arithmetic", (norm_dr, norm_dc), int(first_diff))) #

                            # 2. Identical sequence
                            # 來源：新大腦.pdf - EXT_GM13 Identical check (Page 43)
                            if len(set(s)) == 1 and s[0] != -1: # -1 check might be redundant here #
                                norm_dr = abs(dr_dir) if dc_dir == 0 else dr_dir
                                norm_dc = abs(dc_dir) if dr_dir == 0 else dc_dir #
                                if norm_dr == 1 and norm_dc == 1 and norm_dr * norm_dc < 0:
                                     norm_dr, norm_dc = min(abs(dr_dir),dr_dir), min(abs(dc_dir),dc_dir) if dr_dir != dc_dir else dc_dir
                                found_sequence_signatures.add(("identical", (norm_dr, norm_dc), s[0])) #
                
                current_pval_diversity_count = len(found_sequence_signatures) # 來源：新大腦.pdf (Page 43)
                if current_pval_diversity_count > max_diversity_count_for_cell:
                    max_diversity_count_for_cell = current_pval_diversity_count #
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True
            ) # 來源：新大腦.pdf (Page 43)
            
    return scores * config.weight


# 來源：新大腦.pdf - 20. EXT_GM14_Risk_Assessment_Vec (Page 43) #
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM14強化建議
# Config for this (RiskAssessmentConfig) was defined in PART 2
def EXT_GM14_Risk_Assessment_Vec(
    grid: np.ndarray,
    config: RiskAssessmentConfig,
    request_id: str | None = "N/A_GM14_Risk", #
) -> np.ndarray:
    """
    (GM14 - 風險評估)
    核心規則:評估某個填補動作的潛在「風險」,例如是否會導致後續選擇過少(降低盤面靈活性)。
    目的:偏好那些能保持盤面較高靈活性的填補。低風險=高分數。
    啟發式類型: 盤面狀態評估(未來選擇性)
    輸出詮釋:分數越高表示填入該數字後,盤面剩餘的合法填補選項越多,風險越低。
    來源：新大腦.pdf - EXT_GM14_Risk_Assessment_Vec (Page 43-44)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM14"
    logger.debug(
        f"Executing EXT_GM14_Risk_Assessment_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape #
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 44)
        return scores

    initial_potential_numbers = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # Set[int]
    # 來源：新大腦.pdf (Page 44)
    if not initial_potential_numbers: # 來源：新大腦.pdf (Page 44)
        # If no numbers can be placed initially, all empty cells might be considered max risk (score 0)
        # or neutral (0.5).
        # PDF returns scores (which is zeros). #
        return scores 

    # Heuristic for normalization
    # 來源：新大腦.pdf - EXT_GM14 Heuristic max_possible_options (Page 44)
    # Max possible subsequent_legal_moves is roughly rows*cols (if board almost empty)
    # Max remaining_empty_cells is rows*cols - 1
    # So product can be up to (rows*cols-1)^2
    max_possible_options_heuristic: float
    if config.flexibility_metric_mode == "subsequent_moves":
        max_possible_options_heuristic = float(rows * cols) # Max unique numbers
    else: # "product_moves_empty_cells"
        max_possible_options_heuristic = float((rows * cols -1) * (rows * cols -1)) if rows*cols >1 else 1.0 #
    
    if max_possible_options_heuristic <=0 : max_possible_options_heuristic = 1.0


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 44)
                continue

            max_flexibility_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 44) #
            
            # Iterate through numbers that could be placed at (r_idx, c_idx)
            # This means we should use `initial_potential_numbers` for p_val
            # The PDF note: "p_val in initial_potential_numbers: # Only try values that are currently legal for the original grid"
            # This is correct. #
            evaluated_any_pval = False #
            for p_val in initial_potential_numbers: # p_val is a number that could be placed on the original grid
                                                    # We are evaluating placing it at (r_idx, c_idx)
                evaluated_any_pval = True #
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val # Place p_val

                # Calculate flexibility after this placement
                # 來源：新大腦.pdf - EXT_GM14 Calculate flexibility (Page 44)
                remaining_empty_cells = float(np.count_nonzero(temp_grid == -1)) #
                subsequent_legal_moves_set = BoardAnalyzerUtils.get_legal_values_for_placement(temp_grid)
                num_subsequent_legal_moves = float(len(subsequent_legal_moves_set))

                current_flexibility: float
                if config.flexibility_metric_mode == "subsequent_moves":
                    current_flexibility = num_subsequent_legal_moves # 來源：新大腦.pdf (Page 45) #
                else: # "product_moves_empty_cells"
                    # 來源：新大腦.pdf - EXT_GM14 product metric (Page 45)
                    current_flexibility = remaining_empty_cells * num_subsequent_legal_moves
                
                if current_flexibility > max_flexibility_score_for_cell: # 來源：新大腦.pdf (Page 45) #
                    max_flexibility_score_for_cell = current_flexibility
            
            if not evaluated_any_pval: # Should only happen if initial_potential_numbers was empty
                scores[r_idx,c_idx] = 0.0 # Or some other low score
            else: #
                # 來源：新大腦.pdf - EXT_GM14 Normalization (Page 45)
                # The PDF has `current_max_heuristic_flex = float(rows*cols -1)` which is for subsequent_legal_moves metric.
                # This needs to adapt to the chosen metric. #
                current_max_heuristic_to_use = max_possible_options_heuristic
                if config.flexibility_metric_mode == "subsequent_moves":
                     current_max_heuristic_to_use = float(rows*cols -1) if rows*cols >1 else 1.0 # Max legal after 1 placement
                     if current_max_heuristic_to_use <= 0 : current_max_heuristic_to_use = 1.0
                # else: current_max_heuristic_to_use remains max_possible_options_heuristic which is (R*C-1)^2 #
                
                scores[r_idx, c_idx] = MathUtils.normalize_value(
                    max_flexibility_score_for_cell, 0, current_max_heuristic_to_use, clamp=True
                )

    return scores * config.weight
    # 來源：新大腦.pdf - 21. EXT_GM15_Information_Gain_Vec (Page 45) #
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM15強化建議
# Config for this (InformationGainConfig) was defined in PART 2
def EXT_GM15_Information_Gain_Vec(
    grid: np.ndarray,
    config: InformationGainConfig,
    request_id: str | None = "N/A_GM15_InfoGain", #
) -> np.ndarray:
    """
    (GM15-資訊增益評估)
    核心規則:評估填入數字後,對盤面整體結構「有序性」的提升(例如,熵的降低)。
    目的:偏好那些能使盤面狀態更「確定」或「有序」的填補。
    啟發式類型:資訊理論啟發(基於全局熵變)
    輸出詮釋:分數越高表示填入該數字後,盤面整體熵降低得越多(即資訊增益越大,盤面越有序)。
    來源：新大腦.pdf - EXT_GM15_Information_Gain_Vec (Page 45)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM15"
    logger.debug(
        f"Executing EXT_GM15_Information_Gain_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape #
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 45)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 45)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 45)
        return scores

    # Calculate entropy of the initial grid
    # 來源：新大腦.pdf - EXT_GM15 initial_grid_values (Page 45-46)
    if config.entropy_scope == "global_full":
        initial_grid_values_for_entropy = [int(val) for val in grid.flatten()] # -1 is a symbol #
    else: # "global_filled_only"
        initial_grid_values_for_entropy = [int(val) for val in grid.flatten() if val != -1]
        if not initial_grid_values_for_entropy: # Handle case of all empty grid for filled_only
            initial_grid_values_for_entropy.append(0) # Add a dummy value to avoid empty list for entropy


    entropy_before = MathUtils.get_entropy(initial_grid_values_for_entropy) # 來源：新大腦.pdf (Page 46)

    # Max possible entropy for normalization (log2 of number of symbols: 1 to R*C plus -1 if global_full) #
    # 來源：新大腦.pdf - EXT_GM15 max_possible_entropy_change (Page 46)
    num_symbols_for_max_entropy: int
    if config.entropy_scope == "global_full":
        num_symbols_for_max_entropy = rows * cols + 1 # Numbers 1 to R*C, plus -1
    else: # "global_filled_only"
        num_symbols_for_max_entropy = rows * cols # Numbers 1 to R*C
        if num_symbols_for_max_entropy == 0 : num_symbols_for_max_entropy = 1 # Avoid log2(0) for empty grid

    max_possible_entropy_change = math.log2(num_symbols_for_max_entropy) if num_symbols_for_max_entropy > 1 else 1.0 #
    if max_possible_entropy_change <= 0: max_possible_entropy_change = 1.0 # 來源：新大腦.pdf (Page 46)


    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 46)
                continue

            max_entropy_reduction_for_cell: float = -float('inf') # We want to maximize reduction # 來源：新大腦.pdf (Page 46) #
            
            evaluated_at_least_one_pval = False
            for p_val in potential_numbers_to_place:
                evaluated_at_least_one_pval = True
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val #

                if config.entropy_scope == "global_full":
                    temp_grid_values_for_entropy = [int(val) for val in temp_grid.flatten()]
                else: # "global_filled_only"
                    temp_grid_values_for_entropy = [int(val) for val in temp_grid.flatten() if val != -1]
                    if not temp_grid_values_for_entropy: #
                         temp_grid_values_for_entropy.append(0)


                entropy_after = MathUtils.get_entropy(temp_grid_values_for_entropy) # 來源：新大腦.pdf (Page 46)
                entropy_reduction = entropy_before - entropy_after  # Higher reduction is better # 來源：新大腦.pdf (Page 46)

                if entropy_reduction > max_entropy_reduction_for_cell: # 來源：新大腦.pdf (Page 46) #
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if not evaluated_at_least_one_pval : max_entropy_reduction_for_cell = 0.0 # No legal moves for this cell (should not happen if loop runs)
            elif max_entropy_reduction_for_cell == -float('inf'): max_entropy_reduction_for_cell = 0.0 # 來源：新大腦.pdf (Page 46) #


            # Normalize the reduction. Min reduction can be negative (entropy increases). Max can be entropy_before.
            # Or normalize against max_possible_entropy_change.
            # Score will be higher for positive reductions. #
            # Range of reduction is roughly [-max_possible_entropy_change, max_possible_entropy_change]
            # PDF: MathUtils.normalize_value(max_entropy_reduction_for_cell, 0, max_possible_entropy_change, clamp=True)
            # This normalization clamps negative reductions (entropy increase) to 0.
            # 來源：新大腦.pdf - EXT_GM15 Normalization (Page 46)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_entropy_reduction_for_cell,  #
                0, # Min desired score for reduction (no gain or entropy increase)
                max_possible_entropy_change, # Max possible gain
                clamp=True
            )
            # Clamping at 0 if it increases entropy. (Handled by normalize_value if min_val=0) #
            # 來源：新大腦.pdf (Page 46)
            
    return scores * config.weight


# 來源：新大腦.pdf - 22. EXT_GM16_Harmonic_Centrality_Vec (Page 46)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM16強化建議
# Config for this (HarmonicCentralityConfig) was defined in PART 2
def EXT_GM16_Harmonic_Centrality_Vec(
    grid: np.ndarray,
    config: HarmonicCentralityConfig,
    request_id: str | None = "N/A_GM16_HarmonicCent",
) -> np.ndarray:
    """
    (GM16 - 調和中心性)
    核心規則:應用圖論中的調和中心性概念,評估盤面上各空格節點的重要性。調和中心性是一個節點到所有其他節點距離倒數的總和。
    目的:偏好那些在盤面「網絡」中更具中心性的空格。
    啟發式類型: 圖論中心性 #
    輸出詮釋:分數越高表示該空格在圖結構中越「中心」(平均而言離其他格子越近)。
    來源：新大腦.pdf - EXT_GM16_Harmonic_Centrality_Vec (Page 46-47)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM16"
    logger.debug(
        f"Executing EXT_GM16_Harmonic_Centrality_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    # Needs more than 1 cell # 來源：新大腦.pdf (Page 47) #
    if rows == 0 or cols == 0 or (rows * cols) <= 1:
        return scores

    # Max possible harmonic centrality (heuristic): if a cell is at distance 1 from all N-1 other cells.
    # Max_HC = (rows*cols - 1) * (1/1) #
    # 來源：新大腦.pdf - EXT_GM16 max_hc_heuristic (Page 47)
    max_hc_heuristic = float(rows * cols - 1)
    if max_hc_heuristic <= 0: max_hc_heuristic = 1.0 # 來源：新大腦.pdf (Page 47)

    for r_eval in range(rows):
        for c_eval in range(cols):
            # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM16 節點的定義 (node_definition)
            # Original PDF scores only empty cells.
            if config.node_definition == "empty_cells_only" and grid[r_eval, c_eval] != -1: #
                continue
            if config.node_definition == "filled_cells_only" and grid[r_eval, c_eval] == -1:
                continue
            # if "all_cells", no filter here based on cell content.
            current_harmonic_centrality: float = 0.0 # 來源：新大腦.pdf (Page 47) #
            num_other_nodes_considered = 0

            for r_other in range(rows):
                for c_other in range(cols):
                    if r_eval == r_other and c_eval == c_other: # 來源：新大腦.pdf (Page 47)
                        continue #
                    
                    # Filter other_nodes based on config
                    if config.node_definition == "empty_cells_only" and grid[r_other, c_other] != -1:
                        continue #
                    if config.node_definition == "filled_cells_only" and grid[r_other, c_other] == -1:
                        continue

                    # Using Manhattan distance as grid distance
                    # 來源：新大腦.pdf - EXT_GM16 Manhattan distance (Page 47) #
                    dist = MathUtils.manhattan_distance((r_eval, c_eval), (r_other, c_other))
                    if dist > 0:
                        current_harmonic_centrality += 1.0 / dist
                    num_other_nodes_considered +=1 #
            
            if num_other_nodes_considered == 0: # Only one cell considered, or no valid other_nodes based on filter
                # 來源：新大腦.pdf (Page 47)
                scores[r_eval, c_eval] = 0.0
            else:
                # Normalization can be tricky. Using the heuristic max. #
                # 來源：新大腦.pdf - EXT_GM16 Normalization (Page 48)
                scores[r_eval, c_eval] = MathUtils.normalize_value(
                    current_harmonic_centrality, 0, max_hc_heuristic, clamp=True
                )
    
    # If we only scored specific cells (e.g. empty_cells_only), other cells remain 0.
    return scores * config.weight

# 來源：新大腦.pdf - 23. EXT_GM17_Entropy_Minimization_Vec (Page 48) #
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM17強化建議
# Config for this (LocalEntropyMinimizationConfig) was defined in PART 2
def EXT_GM17_Entropy_Minimization_Vec(
    grid: np.ndarray,
    config: LocalEntropyMinimizationConfig,
    request_id: str | None = "N/A_GM17_LocalEntropy",
) -> np.ndarray:
    """
    (GM17 - 局部熵最小化)
    核心規則:評估填入數字後,盤面局部鄰域「熵」(無序度)的降低程度。
    目的:偏好那些能使其直接周圍環境更有規律、更「有序」的填補。
    啟發式類型:資訊理論啟發(基於局部熵變)
    輸出詮釋:分數越高表示填入該數字後,其局部鄰域的熵降低得越多(局部更有序)。
    來源：新大腦.pdf - EXT_GM17_Entropy_Minimization_Vec (Page 48)
    """
    if not config.enabled: #
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM17"
    logger.debug(
        f"Executing EXT_GM17_Entropy_Minimization_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 48)
        return scores

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 48) #
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 48)
        return scores

    radius = config.radius # 來源：新大腦.pdf (Page 48)
    # Max entropy change in a local neighborhood of size N_hood is log2(N_hood)
    # 來源：新大腦.pdf - EXT_GM17 max_local_entropy_change (Page 48)
    num_cells_in_neighborhood = (2 * radius + 1)**2 # Including center
    max_local_entropy_change = math.log2(num_cells_in_neighborhood) if num_cells_in_neighborhood > 1 else 1.0
    if max_local_entropy_change <= 0: max_local_entropy_change = 1.0 # 來源：新大腦.pdf (Page 48)

    # val_func to keep -1 as a distinct symbol for entropy calculation #
    # 來源：新大腦.pdf - EXT_GM17 val_func_for_entropy (Page 49)
    def val_func_for_entropy(x_val: int) -> int: return int(x_val)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 48)
                continue

            # Get all values in radius around (r_idx,c_idx), including (r_idx,c_idx) itself. #
            # Entropy before (with (r_idx,c_idx) as empty, i.e., -1) #
            # 來源：新大腦.pdf - EXT_GM17 values_before_placement_local (Page 49)
            values_before_placement_local = BoardAnalyzerUtils.get_neighborhood_values(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                val_func=val_func_for_entropy, include_center=True
            )
            entropy_before_local = MathUtils.get_entropy(values_before_placement_local) # 來源：新大腦.pdf (Page 49) #

            max_entropy_reduction_for_cell: float = -float('inf') # 來源：新大腦.pdf (Page 49)
            evaluated_at_least_one_pval = False
            for p_val in potential_numbers_to_place:
                evaluated_at_least_one_pval = True
                temp_grid_local_place = grid.copy() # Create a fresh copy for each p_val
                temp_grid_local_place[r_idx, c_idx] = p_val # 來源：新大腦.pdf (Page 50) #
                
                values_after_placement_local = BoardAnalyzerUtils.get_neighborhood_values(
                    temp_grid_local_place, r_idx, c_idx, radius=radius, eight_connectivity=True,
                    val_func=val_func_for_entropy, include_center=True #
                ) # 來源：新大腦.pdf (Page 50)
                entropy_after_local = MathUtils.get_entropy(values_after_placement_local) # 來源：新大腦.pdf (Page 50)
                
                entropy_reduction = entropy_before_local - entropy_after_local # 來源：新大腦.pdf (Page 50)
                if entropy_reduction > max_entropy_reduction_for_cell: #
                    max_entropy_reduction_for_cell = entropy_reduction
            
            if not evaluated_at_least_one_pval : max_entropy_reduction_for_cell = 0.0
            elif max_entropy_reduction_for_cell == -float('inf'): max_entropy_reduction_for_cell = 0.0 # 來源：新大腦.pdf (Page 50) #


            # Normalize the reduction. Max possible reduction is entropy_before_local,
            # or theoretically max_local_entropy_change if going from max chaos to perfect order.
            # PDF normalizes against max_local_entropy_change. #
            # 來源：新大腦.pdf - EXT_GM17 Normalization (Page 50)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_entropy_reduction_for_cell, 
                0, # Min desired score for reduction (no gain or entropy increase)
                max_local_entropy_change, 
                clamp=True #
            )
            
    return scores * config.weight


# 來源：新大腦.pdf - 24. EXT_GM18_RL_Value_Est_Vec (Page 50)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM18強化建議
# Config for this (RLValueEstimationConfig) was defined in PART 2
def EXT_GM18_RL_Value_Est_Vec(
    grid: np.ndarray,
    config: RLValueEstimationConfig,
    request_id: str | None = "N/A_GM18_RL_Est",
) -> np.ndarray:
    """
    (GM18-類強化學習價值估計)
    核心規則:基於一組預定義的「理想特徴」來評估某個填補動作的啟發式長期潜在價值。此為簡化版,模擬從歷史數據學習到的偏好。
    目的:偏好那些能夠使盤面展現更多理想特徵(如形成特定序列、達到特定盤面密度等)的填補。
    啟發式類型:狀態價值啟發(基於盤面特徵計數)
    輸出詮釋:分數越高表示填入該數字後,盤面呈現的理想特徵越多,預期長期回報越大。
    來源：新大腦.pdf - EXT_GM18_RL_Value_Est_Vec (Page 50) #
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM18"
    logger.debug(
        f"Executing EXT_GM18_RL_Value_Est_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 50)
        return scores #

    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 50)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 51)
        return scores

    FEATURE_WEIGHTS = config.feature_weights # 來源：新大腦.pdf (Page 51)
    
    # Heuristic max feature score for normalization
    # 來源：新大腦.pdf - EXT_GM18 max_heuristic_feature_score (Page 51)
    # Roughly: 4 directions * (max score for identical_3 + max score for arithmetic_3) + density + central + edge
    # This is a very rough estimate as features might overlap or not all be achievable. #
    max_heuristic_feature_score = ( #
        4 * (FEATURE_WEIGHTS.get("identical_3", 0.0) + FEATURE_WEIGHTS.get("arithmetic_3", 0.0)) +
        FEATURE_WEIGHTS.get("board_density_factor", 0.0) * 1.0 + # Max density is 1
        FEATURE_WEIGHTS.get("central_control_boost", 0.0) * 1.0 + # Max central boost is 1
        FEATURE_WEIGHTS.get("edge_affinity_boost", 0.0) * 1.0    # Max edge boost is 1
    )
    if max_heuristic_feature_score <= 0: max_heuristic_feature_score = 1.0 # 來源：新大腦.pdf (Page 51)

    center_r_gm18 = (rows - 1) / 2.0 # For central_control_boost # 來源：新大腦.pdf (Page 52) #
    center_c_gm18 = (cols - 1) / 2.0 # 來源：新大腦.pdf (Page 52)
    max_dist_to_center_gm18 = MathUtils.euclidean_distance((0.0,0.0),(center_r_gm18, center_c_gm18)) if rows*cols > 1 else 0.0
    if math.isclose(max_dist_to_center_gm18, 0.0) and (rows > 1 or cols > 1): max_dist_to_center_gm18 = 1.0


    max_min_dist_to_edge_gm18 = float(min((rows - 1) // 2, (cols - 1) // 2)) # 來源：新大腦.pdf (Page 52)
    if max_min_dist_to_edge_gm18 <=0 and (rows >1 or cols >1): max_min_dist_to_edge_gm18 = 0.5 # Avoid div by zero

    for r_idx in range(rows):
        for c_idx in range(cols): #
            if grid[r_idx, c_idx] != -1:  # Only score empty cells # 來源：新大腦.pdf (Page 51)
                continue
            
            max_feature_score_for_cell: float = 0.0 # 來源：新大腦.pdf (Page 51)
            
            for p_val in potential_numbers_to_place: #
                temp_grid = grid.copy()
                temp_grid[r_idx, c_idx] = p_val
                current_features_score: float = 0.0

                # Feature 1 & 2: Lines of 3 (identical or arithmetic) involving p_val
                # 來源：新大腦.pdf - EXT_GM18 Feature 1 & 2 (Page 51) #
                # Logic similar to GM10/GM5 line checking
                line_len_check = 3 # For identical_3 and arithmetic_3
                for dr_line, dc_line in [(0, 1), (1, 0), (1, 1), (1, -1)]: # H, V, D1, D2 # 來源：新大腦.pdf (Page 51)
                    for i_offset in range(line_len_check): # p_val is at index i_offset #
                        line_values: List[int] = []
                        is_valid_line = True
                        # Check if (r_idx, c_idx) is part of this window implicitly checked by offset #
                        
                        for k_in_segment in range(line_len_check):
                            eval_r = r_idx + (k_in_segment - i_offset) * dr_line #
                            eval_c = c_idx + (k_in_segment - i_offset) * dc_line
                            if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                                is_valid_line = False #
                                break
                            line_values.append(int(temp_grid[eval_r, eval_c])) # 來源：新大腦.pdf (Page 52)
                        
                        if is_valid_line and all(v != -1 for v in line_values): # All filled #
                            s = line_values
                            # Identical
                            # 來源：新大腦.pdf - EXT_GM18 Identical check (Page 52) #
                            if len(set(s)) == 1:
                                current_features_score += FEATURE_WEIGHTS.get("identical_3", 0.0)
                            # Arithmetic (non-constant) #
                            # 來源：新大腦.pdf - EXT_GM18 Arithmetic check (Page 52)
                            elif len(s) >= 2 : # Should be true for len 3 #
                                diffs_feat = [s[k+1] - s[k] for k in range(len(s)-1)]
                                if diffs_feat and len(set(diffs_feat)) == 1 and not math.isclose(diffs_feat[0],0):
                                    current_features_score += FEATURE_WEIGHTS.get("arithmetic_3", 0.0) #
                
                # Feature 3: Board density
                # 來源：新大腦.pdf - EXT_GM18 Board density (Page 52)
                num_filled_after_placement = np.count_nonzero(temp_grid != -1) #
                density_after_placement = num_filled_after_placement / (rows * cols) if (rows * cols) > 0 else 0.0
                current_features_score += FEATURE_WEIGHTS.get("board_density_factor", 0.0) * density_after_placement

                # Conceptual Features (based on GM9, GM8 from PDF)
                # 來源：新大腦.pdf - EXT_GM18 Conceptual Features (Page 52)
                if rows > 1 and cols > 1: # Only for grids larger than 1x1 #
                    # Central control boost
                    if FEATURE_WEIGHTS.get("central_control_boost", 0.0) > 0 and max_dist_to_center_gm18 > 1e-6:
                        dist_to_center = MathUtils.euclidean_distance((float(r_idx), float(c_idx)), (center_r_gm18, center_c_gm18)) #
                        current_features_score += FEATURE_WEIGHTS.get("central_control_boost", 0.0) * \
                            (1.0 - MathUtils.normalize_value(dist_to_center, 0, max_dist_to_center_gm18, clamp=True))
                    
                    # Edge affinity boost (if strategy calls for it, assume prefer_edge for boost) #
                    if FEATURE_WEIGHTS.get("edge_affinity_boost", 0.0) > 0 and max_min_dist_to_edge_gm18 > 1e-6 :
                        dist_to_edge = min(r_idx, rows - 1 - r_idx, c_idx, cols - 1 - c_idx)
                        current_features_score += FEATURE_WEIGHTS.get("edge_affinity_boost", 0.0) * \
                            (1.0 - MathUtils.normalize_value(float(dist_to_edge), 0, max_min_dist_to_edge_gm18, clamp=True)) #
                
                if current_features_score > max_feature_score_for_cell:
                    max_feature_score_for_cell = current_features_score #
            
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_feature_score_for_cell, 0, max_heuristic_feature_score, clamp=True
            ) # 來源：新大腦.pdf (Page 52)
            
    return scores * config.weight


# 來源：新大腦.pdf - 25. EXT_GM19_Masked_Number_Skip_Pattern_Vec (Page 53)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM19強化建議
# Config for this (SkipPatternConfig) was defined in PART 2 #
def EXT_GM19_Masked_Number_Skip_Pattern_Vec(
    grid: np.ndarray,
    config: SkipPatternConfig,
    request_id: str | None = "N/A_GM19_SkipPattern", #
) -> np.ndarray:
    """
    (GM19-遮罩數字跳格模式向量)
    核心規則:分析已揭示數字的「跳格模式」(其實際位置與預期基礎位置的偏差),並對符合主導跳格模式的空格進行評分。
    啟發式類型:空間模式匹配(基於全局偏移量)
    輸出詮釋: 分數越高表示該空格若填入特定數字,能與盤面上觀察到的主要「跳格」規律性最為吻合。
    來源：新大腦.pdf - EXT_GM19_Masked_Number_Skip_Pattern_Vec (Page 53)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM19"
    logger.debug(
        f"Executing EXT_GM19_Masked_Number_Skip_Pattern_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) #
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 53)
        return scores

    # 來源：新大腦.pdf - EXT_GM19 revealed_numbers_info (Page 53)
    revealed_numbers_info: List[Dict[str, Any]] = [
        {'value': int(grid[r, c]), 'r': r, 'c': c}
        for r in range(rows) for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0 # Assuming positive numbers
    ] #
    if not revealed_numbers_info: return scores # 來源：新大腦.pdf (Page 53)

    expected_max_number_on_card = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 53)
    
    # Base positions based on scan pattern (default: left-to-right, top-to-bottom)
    # 來源：新大腦.pdf - EXT_GM19 base_positions (Page 53-54)
    # Conceptual: config.base_pattern_definition could alter this.
    # For now, standard scan. #
    base_positions: Dict[int, Tuple[int, int]] = {} 
    for k_val in range(1, expected_max_number_on_card + 1):
        base_r = (k_val - 1) // cols
        base_c = (k_val - 1) % cols
        if base_r < rows: # Ensure base position is within grid dimensions # 來源：新大腦.pdf (Page 54)
            base_positions[k_val] = (base_r, base_c)

    skip_vectors: Dict[int, Tuple[int, int]] = {} # value -> (delta_r, delta_c) # 來源：新大腦.pdf (Page 54) #
    for rn_info in revealed_numbers_info:
        val = rn_info['value']
        if val in base_positions:
            expected_r, expected_c = base_positions[val]
            skip_vectors[val] = (rn_info['r'] - expected_r, rn_info['c'] - expected_c)
    
    if not skip_vectors: return scores # 來源：新大腦.pdf (Page 54)

    # Determine dominant skip patterns and their strength
    # 來源：新大腦.pdf - EXT_GM19 dominant_skip_patterns_strength (Page 54) #
    dominant_skip_patterns_strength: Dict[Tuple[int, int], float] = {}
    skip_vector_tuples_list = list(skip_vectors.values())
    if not skip_vector_tuples_list: return scores # Should be caught by `if not skip_vectors`

    counts = Counter(skip_vector_tuples_list)
    # 來源：新大腦.pdf - EXT_GM19 min_occurrences_for_pattern (Page 54)
    # PDF: max(1, int(len(skip_vector_tuples_list) * 0.05))
    min_occurrences_for_pattern = max(1, int(len(skip_vector_tuples_list) * config.min_occurrences_for_pattern_factor))
    
    for skip_vec_tuple, count_val in counts.most_common(): # 來源：新大腦.pdf (Page 54)
        if count_val >= min_occurrences_for_pattern:
            # Strength could simply be normalized count #
            # 來源：新大腦.pdf - EXT_GM19 pattern_strength (Page 54)
            pattern_strength = MathUtils.normalize_value(
                float(count_val),
                float(min_occurrences_for_pattern), # Min for a pattern to be considered
                float(len(skip_vector_tuples_list)), # Max possible occurrences (if all same pattern) #
                clamp=True
            )
            dominant_skip_patterns_strength[skip_vec_tuple] = pattern_strength
        else: # Since most_common is sorted
            break # 來源：新大腦.pdf (Page 54) #
            
    if not dominant_skip_patterns_strength: return scores # 來源：新大腦.pdf (Page 54) #

    potential_numbers_to_place_set = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 54)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # 來源：新大腦.pdf (Page 54)
            
            cell_max_pattern_score: float = 0.0 # 來源：新大腦.pdf (Page 54)
            for p_val_test in potential_numbers_to_place_set:
                if p_val_test not in base_positions: continue # 來源：新大腦.pdf (Page 54) #
                
                base_r_test, base_c_test = base_positions[p_val_test]
                for current_skip_pattern, pattern_str in dominant_skip_patterns_strength.items():
                    skip_dr, skip_dc = current_skip_pattern #
                    predicted_r = base_r_test + skip_dr
                    predicted_c = base_c_test + skip_dc

                    if predicted_r == r_idx and predicted_c == c_idx: # Cell matches pattern prediction for p_val_test
                        # 來源：新大腦.pdf (Page 54-55) #
                        current_score_fit = pattern_str # Score is strength of the pattern it fits
                        if current_score_fit > cell_max_pattern_score:
                            cell_max_pattern_score = current_score_fit #
            
            scores[r_idx, c_idx] = cell_max_pattern_score # Max score if multiple patterns/values fit this cell
            # 來源：新大腦.pdf (Page 55)

    return scores * config.weight


# 來源：新大腦.pdf - 26. EXT_GM20_Skip_Pattern_Confidence_Vec (Page 55)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM20強化建議
# Config for this (SkipPatternConfidenceConfig) was defined in PART 2 #
def EXT_GM20_Skip_Pattern_Confidence_Vec(
    grid: np.ndarray,
    config: SkipPatternConfidenceConfig,
    request_id: str | None = "N/A_GM20_SkipConf", #
) -> np.ndarray:
    """
    (GM20-跳格模式信心度/規律性增強)
    核心規則:評估在空格填入數字是否能增強或完成已觀察到的全局跳格規律性, 特別是當這個填補能使遵循跳格模式的數字序列更完整或更具算術規律性時。
    啟發式類型:序列完成與模式確認(基於全局偏移量)
    輸出詮釋:分數越高表示填入該數字不僅符合跳格模式的幾何位置,且能使該模式下的數字序列在算術/序列意義上更為「自信」或「完整」。
    來源：新大腦.pdf - EXT_GM20_Skip_Pattern_Confidence_Vec (Page 55)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM20"
    logger.debug(
        f"Executing EXT_GM20_Skip_Pattern_Confidence_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float) #
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 55)
        return scores

    # --- Initial Pattern Analysis (simplified from GM19, can be refactored into a shared utility) ---
    # 來源：新大腦.pdf - EXT_GM20 Initial Pattern Analysis (Page 55-56)
    revealed_numbers_info_gm20: List[Dict[str, Any]] = [] # 來源：新大腦.pdf (Page 55)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1 and grid[r, c] > 0: # 來源：新大腦.pdf (Page 56) #
                revealed_numbers_info_gm20.append({'value': int(grid[r, c]), 'r': r, 'c': c})
    if not revealed_numbers_info_gm20: return scores # 來源：新大腦.pdf (Page 56)

    expected_max_num_gm20 = BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions((rows, cols)) # 來源：新大腦.pdf (Page 56)
    base_pos_gm20: Dict[int, Tuple[int, int]] = { # 來源：新大腦.pdf (Page 56) #
        k: ((k - 1) // cols, (k - 1) % cols) for k in range(1, expected_max_num_gm20 + 1) if ((k - 1) // cols) < rows #
    }
    skip_vecs_initial_gm20: Dict[int, Tuple[int, int]] = {} # 來源：新大腦.pdf (Page 56)
    for rn in revealed_numbers_info_gm20:
        val = rn['value']
        if val in base_pos_gm20:
            skip_vecs_initial_gm20[val] = (rn['r'] - base_pos_gm20[val][0], rn['c'] - base_pos_gm20[val][1])

    dominant_patterns_details_gm20: List[Dict[str, Any]] = [] # List of {'skip':(dr,dc), 'values':[sorted_values], 'strength':float}
    # 來源：新大腦.pdf (Page 56) #
    if skip_vecs_initial_gm20:
        skip_tuples_list_gm20 = list(skip_vecs_initial_gm20.values()) #
        if not skip_tuples_list_gm20 : return scores # Defensive check
        counts_gm20 = Counter(skip_tuples_list_gm20)
        min_occ_gm20 = max(1, int(len(skip_tuples_list_gm20) * config.min_occurrences_for_pattern_factor_gm20)) # 來源：新大腦.pdf (Page 56)
        
        for skip_v, count_v in counts_gm20.most_common(): # 來源：新大腦.pdf (Page 56)
            if count_v >= min_occ_gm20:
                pattern_vals = sorted([val for val, sv_tuple in skip_vecs_initial_gm20.items() if sv_tuple == skip_v]) #
                p_strength = MathUtils.normalize_value(
                    float(count_v), float(min_occ_gm20), float(len(skip_tuples_list_gm20)), clamp=True
                )
                dominant_patterns_details_gm20.append({'skip': skip_v, 'values': pattern_vals, 'strength': p_strength})
            else: #
                break # 來源：新大腦.pdf (Page 56)
    # --- End Initial Pattern Analysis ---
    if not dominant_patterns_details_gm20: return scores # 來源：新大腦.pdf (Page 56)

    potential_nums_to_place_gm20 = BoardAnalyzerUtils.get_legal_values_for_placement(grid) # 來源：新大腦.pdf (Page 56)

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: continue # 來源：新大腦.pdf (Page 56)
            
            max_confidence_score_for_cell_gm20: float = 0.0 # 來源：新大腦.pdf (Page 56) #
            for p_val_test in potential_nums_to_place_gm20:
                if p_val_test not in base_pos_gm20: continue # 來源：新大腦.pdf (Page 56)
                
                base_r_t, base_c_t = base_pos_gm20[p_val_test] #
                current_max_conf_for_pval: float = 0.0 # 來源：新大腦.pdf (Page 56)

                for pattern_detail in dominant_patterns_details_gm20:
                    pat_skip_dr, pat_skip_dc = pattern_detail['skip']
                    pat_existing_vals = pattern_detail['values']  # sorted list
                    pat_strength = pattern_detail['strength'] #

                    predicted_r_for_pval = base_r_t + pat_skip_dr # 來源：新大腦.pdf (Page 57)
                    predicted_c_for_pval = base_c_t + pat_skip_dc # 來源：新大腦.pdf (Page 57)

                    if predicted_r_for_pval == r_idx and predicted_c_for_pval == c_idx:  # Geometrically fits #
                        enhancement_factor = 0.5  # Base for geometric fit related to pattern strength
                                                # (PDF has 0.5, but this might mean 0.5 * pat_strength)
                                                # Let's consider it a multiplier to pat_strength later. #
                                                # Or, it's an additive factor to a base score of pat_strength.
                                                # PDF: current_conf = pat_strength * enhancement_factor.
                                                # Let's use this. #
                                                # So, if only geometric fit, enhancement_factor = 1.0 for base.
                        current_enhancement_factor = 1.0 # Base for geometric fit #

                        # Check for arithmetic sequence enhancement
                        # 來源：新大腦.pdf - EXT_GM20 Arithmetic sequence enhancement (Page 57)
                        if len(pat_existing_vals) >= 1: # Need at least one existing number #
                            temp_sequence_with_pval = sorted(pat_existing_vals + [p_val_test])
                            if len(temp_sequence_with_pval) >= 2:
                                diffs_in_temp_seq = np.diff(temp_sequence_with_pval) # diff gives array #
                                if len(diffs_in_temp_seq) > 0:
                                    is_arithmetic_now = len(set(diffs_in_temp_seq)) == 1 # All diffs same
                                    first_diff = diffs_in_temp_seq[0] #
                                    
                                    if is_arithmetic_now and not math.isclose(first_diff, 0): # It forms a new, consistent arithmetic sequence #
                                        # 來源：新大腦.pdf (Page 57)
                                        current_enhancement_factor += config.arithmetic_enhancement_bonus 
                                        
                                        # Bonus if p_val_test is between min/max of pat_existing_vals (fills internal gap) #
                                        # 來源：新大腦.pdf (Page 57) #
                                        if len(pat_existing_vals) >=1: # Check to ensure min/max are valid #
                                            min_existing = min(pat_existing_vals) #
                                            max_existing = max(pat_existing_vals)
                                            if min_existing < p_val_test < max_existing : #
                                                current_enhancement_factor += config.internal_gap_fill_bonus
                        
                        current_conf = pat_strength * current_enhancement_factor #  #
                        if current_conf > current_max_conf_for_pval:
                            current_max_conf_for_pval = current_conf
                
                if current_max_conf_for_pval > max_confidence_score_for_cell_gm20: #
                    max_confidence_score_for_cell_gm20 = current_max_conf_for_pval
            
            # Normalization: max_confidence_score_for_cell_gm20 can be > 1 if enhancement_factor > 1.
            # Max pat_strength is 1. Max enhancement can be 1.0 (base) + 0.4 + 0.1 = 1.5
            # So max_conf can be 1.5. #
            # Normalize to [0,1]
            # 來源：新大腦.pdf - EXT_GM20 Normalization (Page 57)
            scores[r_idx, c_idx] = MathUtils.normalize_value(
                max_confidence_score_for_cell_gm20, 0, 1.0 * (1.0 + config.arithmetic_enhancement_bonus + config.internal_gap_fill_bonus), clamp=True
            ) # Max possible heuristic value for current_max_conf_for_pval

    return scores * config.weight


# === Brain Core Dispatch Area ===
# 來源：新大腦.pdf - Brain Core Dispatch Area (Page 6) & Module Registration (Page 58) #
# Using explicit type for the Callable for better clarity with Pydantic configs
BrainModuleCallableWithConfig = Callable[[np.ndarray, Any, str | None], np.ndarray] # grid, config, request_id #
BrainModuleCallableNoConfig = Callable[[np.ndarray, str | None], np.ndarray] # grid, request_id

REGISTERED_MODULES_BRAIN: Dict[str, BrainModuleCallableWithConfig | BrainModuleCallableNoConfig] = { #
    # Modules from brain1.py
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
    "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
    "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
    "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
    "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec,
    "EXT_GM3_Adv_Connected_Comp_Vec": EXT_GM3_Adv_Connected_Comp_Vec,
    # Modules from brain2.py
    "EXT_GM4_Spatial_Auto_Corr_Vec": EXT_GM4_Spatial_Auto_Corr_Vec,
    "EXT_GM5_Line_Completion_Vec": EXT_GM5_Line_Completion_Vec,
    "EXT_GM6_Symmetry_Potential_Vec": EXT_GM6_Symmetry_Potential_Vec,
    "EXT_GM7_Numeric_Gaps_Vec": EXT_GM7_Numeric_Gaps_Vec,
    "EXT_GM8_Edge_Affinity_Vec": EXT_GM8_Edge_Affinity_Vec,
    "EXT_GM9_Center_Control_Vec": EXT_GM9_Center_Control_Vec,
    "EXT_GM10_Blocking_Value_Vec": EXT_GM10_Blocking_Value_Vec,
    "EXT_GM11_Pair_Correlation_Vec": EXT_GM11_Pair_Correlation_Vec,
    "EXT_GM12_Island_Analysis_Vec": EXT_GM12_Island_Analysis_Vec,
    # Modules defined in this file (brain3.py)
    "EXT_GM13_Sequence_Diversity_Vec": EXT_GM13_Sequence_Diversity_Vec,
    "EXT_GM14_Risk_Assessment_Vec": EXT_GM14_Risk_Assessment_Vec,
    "EXT_GM15_Information_Gain_Vec": EXT_GM15_Information_Gain_Vec,
    "EXT_GM16_Harmonic_Centrality_Vec": EXT_GM16_Harmonic_Centrality_Vec,
    "EXT_GM17_Entropy_Minimization_Vec": EXT_GM17_Entropy_Minimization_Vec,
    "EXT_GM18_RL_Value_Est_Vec": EXT_GM18_RL_Value_Est_Vec,
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": EXT_GM19_Masked_Number_Skip_Pattern_Vec,
    "EXT_GM20_Skip_Pattern_Confidence_Vec": EXT_GM20_Skip_Pattern_Confidence_Vec,
}

# Default Pydantic configurations for each module that uses one.
# These would typically be loaded from analyzer_config or a central config store. #
DEFAULT_MODULE_CONFIGS: Dict[str, BaseModel] = { #
    # Configs from brain1.py
    "EXT_A2_Weighted_Proximity_Vec": WeightedProximityConfig(),
    "EXT_M3_Local_Heterogeneity_Vec": LocalHeterogeneityConfig(),
    "EXT_D3_Potential_Field_Vec": PotentialFieldConfig(),
    "EXT_F10_Discontinuity_Vec": DiscontinuityRepairConfig(),
    "EXT_P7_Pathfinding_Value_Vec": PathfindingValueConfig(),
    "EXT_R5_Resource_Control_Vec": ResourceControlConfig(),
    "EXT_GM1_Row_Control_Vec": LineControlConfig(),
    "EXT_GM2_Col_Flow_Vec": LineControlConfig(),
    "EXT_GM3_Adv_Connected_Comp_Vec": ConnectedComponentConfig(),
    # Configs from brain2.py
    "EXT_GM4_Spatial_Auto_Corr_Vec": SpatialAutocorrelationConfig(),
    "EXT_GM5_Line_Completion_Vec": LineCompletionConfig(),
    "EXT_GM6_Symmetry_Potential_Vec": SymmetryPotentialConfig(),
    "EXT_GM7_Numeric_Gaps_Vec": NumericGapsConfig(),
    "EXT_GM8_Edge_Affinity_Vec": EdgeAffinityConfig(),
    "EXT_GM9_Center_Control_Vec": CenterControlConfig(),
    "EXT_GM10_Blocking_Value_Vec": BlockingValueConfigBrain2(), # Use the one from brain2
    "EXT_GM11_Pair_Correlation_Vec": PairCorrelationConfig(),
    "EXT_GM12_Island_Analysis_Vec": IslandAnalysisConfig(),
    # Configs defined in this file (brain3.py) #
    "EXT_GM13_Sequence_Diversity_Vec": SequenceDiversityConfig(),
    "EXT_GM14_Risk_Assessment_Vec": RiskAssessmentConfig(),
    "EXT_GM15_Information_Gain_Vec": InformationGainConfig(),
    "EXT_GM16_Harmonic_Centrality_Vec": HarmonicCentralityConfig(),
    "EXT_GM17_Entropy_Minimization_Vec": LocalEntropyMinimizationConfig(),
    "EXT_GM18_RL_Value_Est_Vec": RLValueEstimationConfig(),
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": SkipPatternConfig(),
    "EXT_GM20_Skip_Pattern_Confidence_Vec": SkipPatternConfidenceConfig(),
}


def get_module_score(
    module_name: str, grid: np.ndarray, config_override: BaseModel | None = None, request_id: str | None = None #
) -> np.ndarray:
    """
    Retrieves and executes a specific scoring module from the registry.
    Args:
        module_name: The registered name of the module to execute.
        grid: The input numpy array representing the game board.
        config_override: Optional Pydantic configuration object to override default for the module.
        request_id: Optional request ID for logging.
    Returns:
        A numpy array containing the scores for each cell, as computed by the module. #
        Returns a zero array of the same shape if the module is not found or an error occurs.
    來源：新大腦.pdf - get_module_score (Page 6)
    Enhanced to use config_override or default config.
    """
    effective_request_id = request_id if request_id else f"N/A_brain_dispatch_{module_name}"
    
    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error(
            f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.", #
            extra={"request_id": effective_request_id},
        )
        rows, cols = grid.shape if grid.ndim == 2 else (0,0)
        return np.zeros((rows, cols), dtype=float)

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    
    # Determine config: use override if provided, else default for that module
    actual_config = config_override if config_override is not None else DEFAULT_MODULE_CONFIGS.get(module_name) #

    if actual_config is None and module_name in DEFAULT_MODULE_CONFIGS: # Should not happen if DEFAULT_MODULE_CONFIGS is complete
        logger.warning(f"Default config not found for module {module_name}, but it expects one. Using base config.", #
                       extra={"request_id": effective_request_id})
        actual_config = BaseModuleConfig() # Fallback, module might fail if it expects specific fields
    
    # Check if module actually expects a config based on its Pydantic config class existence
    # (More robust: inspect function signature, but for now assume if it's in DEFAULT_MODULE_CONFIGS it takes one)

    logger.info(
        f"Executing module: {module_name} with config: {actual_config.model_dump_json(indent=2) if actual_config else 'None'}", #
        extra={"request_id": effective_request_id},
    )
    try:
        if module_name in DEFAULT_MODULE_CONFIGS: # Assumes modules with entry in DEFAULT_MODULE_CONFIGS take a config argument
            if actual_config is None: # Should be caught above
                 raise ValueError(f"Module {module_name} requires a config but none was provided or defaulted correctly.")
            score_grid = module_func(grid, config=actual_config, request_id=effective_request_id) #
        else: 
            # This case is for modules that might not have/need a Pydantic config
            # However, our design makes all of them take one (even if it's just BaseModuleConfig)
            # For safety, if a module is registered but not in DEFAULT_MODULE_CONFIGS, assume it takes no config #
            # This path should ideally not be taken if all modules are consistently defined.
            # Let's assume all our 26 modules will have a config, even if it's just BaseModuleConfig. #
            # score_grid = module_func(grid, request_id=effective_request_id) # Fallback if no config expected #
            # Re-evaluating: All modules are now designed to take a config object.
            # So, if actual_config is still None here, it's an issue. #
            if actual_config is None: #
                 logger.error(f"Internal error: Module {module_name} expected a config, but it's None.", extra={"request_id": effective_request_id})
                 rows, cols = grid.shape if grid.ndim == 2 else (0,0)
                 return np.zeros((rows, cols), dtype=float)
            score_grid = module_func(grid, config=actual_config, request_id=effective_request_id)


        if not isinstance(score_grid, np.ndarray) or score_grid.shape != grid.shape: #
            logger.error(f"Module {module_name} returned invalid score_grid. Shape: {score_grid.shape if isinstance(score_grid, np.ndarray) else type(score_grid)}, Expected: {grid.shape}",
                           extra={"request_id": effective_request_id})
            rows, cols = grid.shape if grid.ndim == 2 else (0,0)
            return np.zeros((rows, cols), dtype=float)

        return score_grid #
    except Exception as e:
        logger.error(
            f"Error executing module {module_name}: {e}",
            exc_info=True,
            extra={"request_id": effective_request_id},
        )
        rows, cols = grid.shape if grid.ndim == 2 else (0,0)
        return np.zeros((rows, cols), dtype=float)


# 來源：新大腦.pdf - Verification (Page 58-60) #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [RID:%(request_id)s]'
    )
    # Add a simple handler that includes request_id if not present in extra
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(lambda record: hasattr(record, 'request_id') or setattr(record, 'request_id', 'direct_run'))


    print("Verifying brain.py structure and all 26 modules...")
    dummy_grid_np = np.array([ #
        [1, 2, -1, 4, 5], 
        [-1, 5, -1, 8, -1], 
        [3, -1, 4, -1, 11],
        [12,13,-1,15,16],
        [-1,18,-1,20,-1]
    ], dtype=int)
    print(f"Created dummy grid (5x5):\n{dummy_grid_np}")

    total_modules = len(REGISTERED_MODULES_BRAIN)
    print(f"\nTotal modules registered: {total_modules}")
    assert total_modules == 26, f"Expected 26 modules, found {total_modules}"

    successful_runs = 0
    failed_modules = [] #

    for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
        print(f"\n--- Testing module {i+1}/{total_modules}: {name} ---")
        specific_config_override = None
        # Example: Override config for a specific module if needed for testing
        # if name == "EXT_A2_Weighted_Proximity_Vec":
        #     specific_config_override = WeightedProximityConfig(radius=1, weight=0.5)
        
        try:
            scores_array = get_module_score(name, dummy_grid_np, config_override=specific_config_override, request_id=f"test_{name}") #
            print(f"Successfully called {name}. Output shape: {scores_array.shape}, dtype: {scores_array.dtype}") #
            if scores_array.shape != dummy_grid_np.shape:
                print(f"ERROR: Shape mismatch for {name}! Expected {dummy_grid_np.shape}, Got {scores_array.shape}")
                failed_modules.append(name + " (shape mismatch)")
                continue
            if scores_array.dtype != float:
                print(f"ERROR: Dtype mismatch for {name}! Expected float, Got {scores_array.dtype}") #
                failed_modules.append(name + " (dtype mismatch)")
                continue
            
            # Print a small sample of scores
            sample_scores = scores_array[0:min(3,scores_array.shape[0]), 0:min(3,scores_array.shape[1])]
            print(f"Sample scores for {name}:\n{sample_scores}") #
            successful_runs += 1

        except Exception as e:
            print(f"ERROR executing module {name}: {e}")
            logger.exception(f"Exception during test of {name}")
            failed_modules.append(name + f" (execution error: {type(e).__name__})")
    
    print("\n--- Verification Summary ---")
    print(f"Successfully ran {successful_runs}/{total_modules} modules.") #
    if failed_modules:
        print("Failed modules:")
        for f_mod in failed_modules:
            print(f"  - {f_mod}")
    else:
        print("All registered modules ran without immediate errors (shape/dtype checks passed).")

    print("\nbrain.py verification complete.")