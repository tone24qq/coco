# brain2.py
# This is a continuation of the optimized brain.py module.
# Ensure that definitions from brain1.py (MathUtils, BoardAnalyzerUtils, BaseModuleConfig, etc.)
# are available in the execution context if these parts are run as separate conceptual files.
# For a single combined file, this will flow naturally.

import numpy as np
import math
from collections import Counter, deque # Counter might not be fully Numba compatible if used inside njit
import logging
from typing import List, Dict, Tuple, Callable, Optional, Any, Set, Union # Added Union

import numba
from numba import njit, prange, typed

from pydantic import BaseModel, Field

# Assuming logger, MathUtils, BoardAnalyzerUtils, and BaseModuleConfig 
# and initial configs are defined as in brain1.py
# (If running as separate files, these would need to be imported or redefined.
# For this response, I'm treating this as a single continuous file split for display.)

# --- Pydantic Config Models for Modules (Continued from brain1.py) ---

class SymmetryPotentialConfig(BaseModuleConfig): # For GM6
    # 來源：新大腦.pdf - EXT_GM6 parameters (Page 27-28)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM6 對稱類型權重
    score_horizontal: float = Field(default=0.7, ge=0.0)
    score_vertical: float = Field(default=0.7, ge=0.0)
    score_point_center: float = Field(default=0.8, ge=0.0)
    score_main_diagonal: float = Field(default=0.6, ge=0.0)
    score_anti_diagonal: float = Field(default=0.6, ge=0.0)
    strict_square_for_diagonal: bool = Field(default=True, description="對角線對稱是否嚴格要求方形棋盤") # 來源：新大腦.pdf (Page 29) [cite: 135, 138] #

class NumericGapsConfig(BaseModuleConfig): # For GM7
    # 來源：新大腦.pdf - EXT_GM7 parameters (Page 29-30)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM7 「間隙」的上下文
    score_arithmetic_1_gap_fill: float = Field(default=0.9, ge=0.0)
    score_arithmetic_generic_mend: float = Field(default=0.7, ge=0.0)
    score_arithmetic_generic_extend: float = Field(default=0.5, ge=0.0)
    # 來源：新大腦.pdf - EXT_GM7 Added: scoring for quality (conceptual) (Page 30)
    enable_quality_enhancement_gm7: bool = Field(default=True) #
    score_gap_fill_high_val_bonus: float = Field(default=0.1, ge=0.0) #
    high_value_threshold_factor_gm7: float = Field(default=0.66, ge=0, le=1)

class EdgeAffinityConfig(BaseModuleConfig): # For GM8
    # 來源：新大腦.pdf - EXT_GM8 parameters (Page 31-32)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM8 affinity_mode 的動態化
    affinity_mode: str = Field(default="prefer_edge", pattern="^(prefer_edge|avoid_edge)$") #
    corner_bonus_prefer: float = Field(default=0.2, ge=0.0) #
    corner_penalty_avoid: float = Field(default=0.2, ge=0.0) #

class CenterControlConfig(BaseModuleConfig): # For GM9
    # 來源：新大腦.pdf - EXT_GM9 parameters (Page 34)
    affinity_mode: str = Field(default="prefer_center", pattern="^(prefer_center|avoid_center)$") #

class BlockingValueConfig(BaseModuleConfig): # For GM10
    # 來源：新大腦.pdf - EXT_GM10 parameters (Page 35-36)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM10 UNDESIRABLE_SEQUENCES 的擴展與學習 #
    undesirable_sequences_list: List[List[int]] = Field(default_factory=lambda: [
        [1, 1, 1], [2, 2, 2] # 來源：新大腦.pdf (Page 36)
    ])
    score_if_safe: float = Field(default=0.9, ge=0.0, le=1.0, description="Score if placement does NOT complete an undesirable pattern.") # 來源：新大腦.pdf (Page 37) #
    score_if_unsafe: float = Field(default=0.1, ge=0.0, le=1.0, description="Score if placement DOES complete an undesirable pattern.") # 來源：新大腦.pdf (Page 37) #
    check_line_length: int = Field(default=3, ge=2, description="Length of lines to check for undesirable patterns.")

class PairCorrelationConfig(BaseModuleConfig): # For GM11
    # 來源：新大腦.pdf - EXT_GM11 parameters (Page 38-39)
    favorable_pairs: Dict[Tuple[int, int], float] = Field(default_factory=lambda: { #
        (3, 7): 0.8, (7, 3): 0.8, (1, 2): 0.6, (2, 1): 0.6, (10,20):0.7, (20,10):0.7
    })

class IslandAnalysisConfig(BaseModuleConfig): # For GM12
    # 來源：新大腦.pdf - EXT_GM12 parameters (Page 40-41)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM12 島嶼特徵的豐富化
    w_size: float = Field(default=0.4, ge=0.0, le=1.0) #
    w_compactness: float = Field(default=0.3, ge=0.0, le=1.0) #
    w_avg_value: float = Field(default=0.3, ge=0.0, le=1.0) #

class SequenceDiversityConfig(BaseModuleConfig): # For GM13
    # 來源：新大腦.pdf - EXT_GM13 parameters (Page 42) #
    short_sequence_len: int = Field(default=3, ge=2) #

class RiskAssessmentConfig(BaseModuleConfig): # For GM14
    # 來源：新大腦.pdf - EXT_GM14 parameters (Page 44)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM14 靈活性度量的複雜化
    flexibility_metric_mode: str = Field(default="subsequent_moves", pattern="^(subsequent_moves|product_moves_empty_cells)$")

class InformationGainConfig(BaseModuleConfig): # For GM15
    # 來源：新大腦.pdf - EXT_GM15 parameters (Page 45-46)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM15 熵計算的對象 #
    entropy_scope: str = Field(default="global_full", pattern="^(global_full|global_filled_only)$", description="熵計算範圍：global_full (含-1), global_filled_only (不含-1)")

class HarmonicCentralityConfig(BaseModuleConfig): # For GM16
    # 來源：新大腦.pdf - EXT_GM16 parameters (Page 47)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM16 節點的定義
    node_definition: str = Field(default="all_cells", pattern="^(all_cells|empty_cells_only|filled_cells_only)$", description="計算調和中心性時考慮的節點類型")

class LocalEntropyMinimizationConfig(BaseModuleConfig): # For GM17
    # 來源：新大腦.pdf - EXT_GM17 parameters (Page 48)
    radius: int = Field(default=1, ge=1, description="局部鄰域半徑")

class RLValueEstimationConfig(BaseModuleConfig): # For GM18
    # 來源：新大腦.pdf - EXT_GM18 parameters (Page 50-51)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM18 特徵庫的擴展與優化 #
    feature_weights: Dict[str, float] = Field(default_factory=lambda: { #
        "identical_3": 1.0,
        "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.1, # 來源：新大腦.pdf (Page 51)
        "edge_affinity_boost": 0.05,   # 來源：新大腦.pdf (Page 52)
    })

class SkipPatternConfig(BaseModuleConfig): # For GM19
    # 來源：新大腦.pdf - EXT_GM19 parameters (Page 53-54)
    min_occurrences_for_pattern_factor: float = Field(default=0.05, ge=0.0, le=1.0, description="形成主導跳格模式所需的最少出現次數（佔總跳格數的比例）") # PDF uses 0.05 of len(skip_vector_tuples_list) [cite: 229] #
    base_pattern_definition: str = Field(default="left_to_right_top_to_bottom", description="理論基礎位置的掃描模式（概念性）")

class SkipPatternConfidenceConfig(BaseModuleConfig): # For GM20
    # 來源：新大腦.pdf - EXT_GM20 parameters (Page 55-56)
    min_occurrences_for_pattern_factor_gm20: float = Field(default=0.05, ge=0.0, le=1.0) 
    # 來源：新大腦.pdf - EXT_GM20 arithmetic sequence enhancement (Page 57)
    arithmetic_enhancement_bonus: float = Field(default=0.4, ge=0.0, description="形成一致等差序列的增強因子") #
    internal_gap_fill_bonus: float = Field(default=0.1, ge=0.0, description="填充內部間隙形成等差序列的額外獎勵") #

# --- Scoring Module Implementations ---

# 來源：新大腦.pdf - 1. EXT_A2_Weighted_Proximity_Vec (Page 7) [cite: 21]
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_A2強化建議
# 來源：给你2025资料在深度建议一次.pdf - EXT_A2 Pydantic配置範例 (Page 2)
@njit(parallel=True)
def EXT_A2_Weighted_Proximity_Vec_numba(
    grid: np.ndarray,
    radius: int,
    value_weight_factor: float,
    distance_decay_factor: float,
    # enable_repulsion: bool, # Repulsion logic needs placed_value, handled by analyzer if used
    # undesirable_pairs_config_keys: numba.typed.List, # Numba-friendly representation
    # undesirable_pairs_config_values: np.ndarray,     # Numba-friendly representation
    max_val_on_grid: float,
    heuristic_max_score_val: float # Renamed from heuristic_max_score to avoid conflict
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0:
        return scores

    for r_idx in prange(rows): #
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: #
                continue

            proximity_score = 0.0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0: # 來源：新大腦.pdf (Page 8) [cite: 21]
                        continue
                    
                    nr, nc = r_idx + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        dist = abs(r_idx - nr) + abs(c_idx - nc) # Manhattan distance #
                        if dist == 0: dist = 1 # Safeguard

                        score_contribution = (
                            grid[nr, nc] * value_weight_factor
                        ) / (dist**distance_decay_factor) # 來源：新大腦.pdf (Page 8) [cite: 22] #
                        proximity_score += score_contribution
            
            if heuristic_max_score_val > 1e-9: # Avoid division by zero or near-zero # 來源：新大腦.pdf (Page 8) [cite: 23]
                # Using MathUtils static methods directly in Numba requires them to be jitted
                # and Numba to correctly handle the class context or pass them as functions.
                # For simplicity, re-implement normalize logic here or ensure MathUtils.normalize_value is Numba-compatible.
                # Assuming MathUtils.normalize_value is Numba jitted as per brain1.py
                scores[r_idx, c_idx] = MathUtils.normalize_value( #
                    proximity_score, 0.0, heuristic_max_score_val, clamp=True
                )
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

def EXT_A2_Weighted_Proximity_Vec(
    grid: np.ndarray,
    config: WeightedProximityConfig, 
    request_id: str | None = "N/A_A2_Proximity", #
) -> np.ndarray:
    """
    (A2-加權鄰近性) [cite: 21]
    核心規則:評估空格周圍已填數字的接近程度及其值的影響。[cite: 21]
    目的:偏好靠近高價值數字或數字密集區域的空格。[cite:21]
    啟發式類型:空間鄰近性 [cite: 21]
    輸出詮釋:分數越高表示鄰近效應越強(受周圍數字的值與密度影響) [cite: 21]
    來源：新大腦.pdf - EXT_A2_Weighted_Proximity_Vec (Page 7)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_A2"
    logger.debug(
        f"Executing EXT_A2_Weighted_Proximity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    ) #

    rows, cols = grid.shape
    if rows == 0 or cols == 0: return np.zeros((rows,cols), dtype=float)

    max_val_on_grid = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(rows, cols)) #
    if max_val_on_grid == 0: max_val_on_grid = 1.0

    num_neighbors_in_radius = (2 * config.radius + 1) ** 2 - 1
    heuristic_max_score = ( #
        num_neighbors_in_radius
        * max_val_on_grid # Max possible value
        * config.value_weight_factor
    ) / (1.0**config.distance_decay_factor) # Min dist is 1
    if heuristic_max_score <= 1e-9: heuristic_max_score = 1.0 #

    # Repulsion logic not directly implemented here as it needs a `some_proposed_val_for_this_cell`
    # This would typically be handled by an `analyzer` layer iterating proposed values.
    
    scores = EXT_A2_Weighted_Proximity_Vec_numba(
        grid,
        config.radius,
        config.value_weight_factor,
        config.distance_decay_factor,
        max_val_on_grid,
        heuristic_max_score
    )
    return scores * config.weight


# 來源：新大腦.pdf - 2. EXT_M3_Local_Heterogeneity_Vec (Page 8)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_M3強化建議
@njit(parallel=True)
def EXT_M3_Local_Heterogeneity_Vec_numba(
    grid: np.ndarray,
    radius: int,
    min_neighbors_for_robust_score: int,
    diversity_metric_code: int, # 0: entropy, 1: gini, 2: unique_count
    all_possible_values_count: int, # num_distinct_symbols
    max_theoretical_diversity_measure_val: float # Renamed
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores
    if all_possible_values_count == 0: return scores #

    for r_idx in prange(rows): #
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1: #
                continue
            
            # Using BoardAnalyzerUtils.get_neighborhood_values_numba (already jitted)
            neighbor_values_typed_list = BoardAnalyzerUtils.get_neighborhood_values_numba(
                grid, r_idx, c_idx, radius=radius, eight_connectivity=True,
                include_center=False,
            ) #

            if len(neighbor_values_typed_list) < min_neighbors_for_robust_score: # 來源：新大腦.pdf (Page 10)
                scores[r_idx, c_idx] = 0.0
                continue

            current_diversity_value: float = 0.0
            if diversity_metric_code == 0: # entropy
                # Numba-fied get_entropy (simplified if values are floats already)
                if not neighbor_values_typed_list:
                    current_diversity_value = 0.0
                else:
                    # Numba-compatible Counter
                    counts_dict = numba.typed.Dict() # type: ignore
                    for item_val in neighbor_values_typed_list:
                        # Numba dicts need explicit typing if keys are not simple
                        # Assuming neighbor_values_typed_list contains floats that can be dict keys
                        int_item_val = int(round(item_val)) # Or handle floats as keys if appropriate
                        counts_dict[int_item_val] = counts_dict.get(int_item_val, 0) + 1
                    
                    total_count_local = len(neighbor_values_typed_list)
                    entropy_local = 0.0
                    for count_local in counts_dict.values():
                        probability_local = count_local / total_count_local
                        if probability_local > 0:
                             entropy_local -= probability_local * math.log2(probability_local)
                    current_diversity_value = entropy_local # 來源：新大腦.pdf (Page 10) [cite: 35]
            
            elif diversity_metric_code == 1: # gini
                if not neighbor_values_typed_list:
                    current_diversity_value = 0.0
                else:
                    counts_dict_gini = numba.typed.Dict() # type: ignore
                    for item_val_gini in neighbor_values_typed_list:
                        int_item_val_gini = int(round(item_val_gini))
                        counts_dict_gini[int_item_val_gini] = counts_dict_gini.get(int_item_val_gini, 0) + 1
                    
                    impurity = 1.0 #
                    len_neighbor_values = len(neighbor_values_typed_list)
                    for count_val_gini in counts_dict_gini.values():
                        prob = count_val_gini / len_neighbor_values
                        impurity -= prob**2
                    current_diversity_value = impurity 
            
            elif diversity_metric_code == 2: # unique_count
                if not neighbor_values_typed_list:
                     current_diversity_value = 0.0
                else:
                    # Numba-compatible unique count
                    unique_set = set()
                    for item_uc in neighbor_values_typed_list:
                        unique_set.add(int(round(item_uc))) # Assuming values are effectively integers
                    
                    current_diversity_value = float(len(unique_set)) #
                    max_possible_unique_in_neighborhood = min(len(neighbor_values_typed_list), all_possible_values_count)
                    if max_possible_unique_in_neighborhood > 0 :
                        current_diversity_value = current_diversity_value / max_possible_unique_in_neighborhood #
                    else:
                        current_diversity_value = 0.0
            
            # For unique_count, max_theoretical_diversity_measure_val should be 1.0 for its already normalized value
            effective_max_diversity = max_theoretical_diversity_measure_val
            if diversity_metric_code == 2: # unique_count mode
                 effective_max_diversity = 1.0


            if effective_max_diversity > 1e-9: #
                normalized_score = current_diversity_value / effective_max_diversity #
                scores[r_idx, c_idx] = MathUtils.normalize_value( #
                    normalized_score, 0.0, 1.0, clamp=True 
                )
            else:
                scores[r_idx, c_idx] = 0.0
    return scores

def EXT_M3_Local_Heterogeneity_Vec(
    grid: np.ndarray,
    config: LocalHeterogeneityConfig, 
    request_id: str | None = "N/A_M3_Heterogeneity", #
) -> np.ndarray:
    """
    (M3 - 局部異質性)
    核心規則:評估空格周圍數字的多樣性。
    目的:偏好周圍數字分佈更隨機、更少重複的空格。
    啟發式類型:分佈統計(基於熵)
    輸出詮釋: 分數越高表示周圍環境的數字異質性越高(熵越大)
    來源：新大腦.pdf - EXT_M3_Local_Heterogeneity_Vec (Page 8-9)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_M3"
    logger.debug(
        f"Executing EXT_M3_Local_Heterogeneity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    if rows == 0 or cols == 0: return np.zeros((rows,cols), dtype=float)

    min_neighbors = config.min_neighbors_for_robust_score
    if rows * cols < 10:
        min_neighbors = max(0, min(config.min_neighbors_for_robust_score, 1)) #

    all_possible_nums_set = BoardAnalyzerUtils.get_all_possible_numbers_for_grid(grid.shape) # 來源：新大腦.pdf (Page 9)
    num_distinct_symbols = len(all_possible_nums_set)
    if num_distinct_symbols == 0 and grid.size > 0 : # Should not happen if grid has valid numbers range
        num_distinct_symbols = rows * cols # Fallback to max possible if set is empty but grid is not

    max_theoretical_diversity: float
    if num_distinct_symbols > 1:
        max_theoretical_diversity = math.log2(num_distinct_symbols) #
    elif num_distinct_symbols == 1:
        max_theoretical_diversity = math.log2(2.0) #
    else: 
        max_theoretical_diversity = 1.0 
    if max_theoretical_diversity <= 1e-9: max_theoretical_diversity = 1.0

    metric_code = {"entropy": 0, "gini": 1, "unique_count": 2}.get(config.diversity_metric, 0)

    scores = EXT_M3_Local_Heterogeneity_Vec_numba(
        grid,
        config.radius,
        min_neighbors,
        metric_code,
        num_distinct_symbols,
        max_theoretical_diversity
    )
    return scores * config.weight


# 來源：新大腦.pdf - 3. EXT_D3_Potential_Field_Vec (Page 10)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_D3強化建議
# Vectorized approach for D3 as suggested
def EXT_D3_Potential_Field_Vec(
    grid: np.ndarray,
    config: PotentialFieldConfig,
    request_id: str | None = "N/A_D3_Potential", #
) -> np.ndarray:
    """
    (D3-位勢場分析) - Optimized with vectorization
    核心規則:將盤面上的數字視為「電荷」,空格則根據其位置的「綜合位勢」來評分。
    目的:偏好位於受高價值數字「吸引」或低價值數字「排斥」(如果設計如此)區域的空格。
    啟發式類型:物理類比(類似靜電場或重力場)
    輸出詮釋:分數越高表示該空格受到周圍數字的正向「位勢影響」越大
    來源：新大腦.pdf - EXT_D3_Potential_Field_Vec (Page 10) [cite: 39]
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_D3"
    logger.debug(
        f"Executing EXT_D3_Potential_Field_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: #
        return scores

    empty_cells_coords = np.argwhere(grid == -1)
    filled_cells_coords = np.argwhere(grid != -1)

    if empty_cells_coords.shape[0] == 0 or filled_cells_coords.shape[0] == 0:
        return scores * config.weight # No interactions possible

    # Get charge values from filled cells
    charge_values = grid[filled_cells_coords[:, 0], filled_cells_coords[:, 1]].astype(np.float64)
    if config.enable_negative_charges:
        for i, coord_idx in enumerate(filled_cells_coords):
            val = grid[coord_idx[0], coord_idx[1]]
            if val in config.negative_charge_map:
                charge_values[i] = config.negative_charge_map[val] #
            elif val <= 0: # Original logic for non-negative charges
                 charge_values[i] = 0 # effectively removing non-positive, non-mapped charges


    # Calculate Manhattan distances between all empty cells and all filled cells
    # cdist returns a matrix: distances[i, j] = distance between empty_cells_coords[i] and filled_cells_coords[j]
    try:
        from scipy.spatial.distance import cdist
        distances = cdist(empty_cells_coords, filled_cells_coords, metric='cityblock')
    except ImportError:
        logger.warning("SciPy not found for EXT_D3, falling back to slower Numba/loop implementation for distances.")
        # Fallback (slower, but Numba-accelerated loop for distance matrix)
        @njit
        def calculate_distance_matrix(empty_coords_n, filled_coords_n):
            dist_matrix = np.empty((empty_coords_n.shape[0], filled_coords_n.shape[0]), dtype=np.float64)
            for i_ec_val in range(empty_coords_n.shape[0]):
                for j_fc_val in range(filled_coords_n.shape[0]):
                    dist_matrix[i_ec_val, j_fc_val] = abs(empty_coords_n[i_ec_val, 0] - filled_coords_n[j_fc_val, 0]) + \
                                                 abs(empty_coords_n[i_ec_val, 1] - filled_coords_n[j_fc_val, 1])
            return dist_matrix
        distances = calculate_distance_matrix(empty_cells_coords, filled_cells_coords)


    distances[distances == 0] = 1e-9 # Avoid division by zero, effectively infinite potential if dist=0
    if config.max_influence_radius > 0:
        distances[distances > config.max_influence_radius] = np.inf # Ignore cells beyond radius

    potential_contributions = charge_values[np.newaxis, :] / (distances ** config.decay_exponent) #
    
    # Sum contributions for each empty cell (sum over filled cells axis)
    total_potentials = np.sum(potential_contributions, axis=1)

    # Normalization
    max_possible_val_on_grid = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(rows, cols)) #
    if max_possible_val_on_grid == 0: max_possible_val_on_grid = 1.0

    num_cells_in_radius_approx = (2 * config.max_influence_radius + 1)**2 - 1 #
    heuristic_max_potential = num_cells_in_radius_approx * (
        max_possible_val_on_grid / (1.0**config.decay_exponent) 
    )
    if heuristic_max_potential <= 1e-9: heuristic_max_potential = 1.0 #

    normalized_potentials = np.empty_like(total_potentials, dtype=np.float64)
    for i in range(total_potentials.shape[0]):
        # Note: If negative charges can make total_potential negative, min_val for normalization might change.
        # Current MathUtils.normalize_value(value, 0, max, clamp) will clamp negative to 0.
        normalized_potentials[i] = MathUtils.normalize_value(total_potentials[i], 0.0, heuristic_max_potential, clamp=True) #

    scores[empty_cells_coords[:, 0], empty_cells_coords[:, 1]] = normalized_potentials
    return scores * config.weight


# 來源：新大腦.pdf - 4. EXT_F10_Discontinuity_Vec (Page 12)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_F10強化建議
# Config for this (DiscontinuityRepairConfig) was defined in brain1.py
@njit(parallel=True)
def EXT_F10_Discontinuity_Vec_numba(
    grid: np.ndarray,
    legal_values_arr: np.ndarray, # Array of legal values
    min_sequence_len_to_score: int,
    allow_gaps_in_sequence: int,
    check_arithmetic: bool,
    check_geometric: bool,
    sequence_quality_weighting: bool,
    high_value_sequence_threshold_factor: float,
    heuristic_max_len_val: float # Renamed
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0 or legal_values_arr.shape[0] == 0: #
        return scores

    max_board_val_f10 = float(rows * cols) # Max value for quality weighting
    if max_board_val_f10 == 0: max_board_val_f10 = 1.0
    
    for r_idx in prange(rows): #
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            max_len_contribution_for_this_cell: float = 0.0 #

            for val_to_try_idx in range(legal_values_arr.shape[0]):
                val_to_try = legal_values_arr[val_to_try_idx]
                temp_grid = grid.copy() # Copy is expensive in Numba loop, but necessary for isolated checks.
                temp_grid[r_idx, c_idx] = val_to_try #
                current_val_max_len: float = 0.0

                # Directions: Row, Col, Diag1, Diag2
                # 1. Row
                row_line = np.array([float(x) if x != -1 else np.nan for x in temp_grid[r_idx, :]], dtype=np.float64)
                sequences_found_row = BoardAnalyzerUtils.find_sequences_in_line_numba(
                    row_line, min_sequence_len_to_score, check_arithmetic, check_geometric, allow_gaps_in_sequence
                )
                for seq_idx in range(len(sequences_found_row)):
                    seq = sequences_found_row[seq_idx]
                    is_val_in_seq = False
                    for item_in_seq_idx in range(len(seq)):
                        if seq[item_in_seq_idx] == val_to_try : is_val_in_seq = True; break
                    if is_val_in_seq:
                        seq_len = float(len(seq)) #
                        if sequence_quality_weighting:
                            seq_sum = 0.0
                            for item_s_idx in range(len(seq)): seq_sum += seq[item_s_idx]
                            avg_val_in_seq = seq_sum / len(seq) if len(seq) > 0 else 0.0 #
                            if max_board_val_f10 > 0 and avg_val_in_seq > (max_board_val_f10 * high_value_sequence_threshold_factor): #
                                seq_len *= 1.2 
                        current_val_max_len = max(current_val_max_len, seq_len)

                # 2. Column
                col_line = np.array([float(x) if x != -1 else np.nan for x in temp_grid[:, c_idx]], dtype=np.float64)
                sequences_found_col = BoardAnalyzerUtils.find_sequences_in_line_numba(
                    col_line, min_sequence_len_to_score, check_arithmetic, check_geometric, allow_gaps_in_sequence
                )
                for seq_idx_c in range(len(sequences_found_col)):
                    seq_c = sequences_found_col[seq_idx_c]
                    is_val_in_seq_c = False
                    for item_in_seq_idx_c in range(len(seq_c)):
                         if seq_c[item_in_seq_idx_c] == val_to_try : is_val_in_seq_c = True; break
                    if is_val_in_seq_c:
                        seq_len_c = float(len(seq_c)) #
                        if sequence_quality_weighting:
                            seq_sum_c = 0.0
                            for item_s_idx_c in range(len(seq_c)): seq_sum_c += seq_c[item_s_idx_c]
                            avg_val_in_seq_c = seq_sum_c / len(seq_c) if len(seq_c) > 0 else 0.0 #
                            if max_board_val_f10 > 0 and avg_val_in_seq_c > (max_board_val_f10 * high_value_sequence_threshold_factor): #
                                seq_len_c *= 1.2
                        current_val_max_len = max(current_val_max_len, seq_len_c)

                # 3. Diagonal 1 (\)
                diag1_offset = c_idx - r_idx
                diag1_elements = numba.typed.List() # type: numba.typed.List[np.float_]
                for i_d1 in range(max(rows,cols)): # Iterate enough to capture full diagonal
                    r_d1, c_d1 = i_d1, i_d1 + diag1_offset
                    if 0 <= r_d1 < rows and 0 <= c_d1 < cols:
                        val_d1 = temp_grid[r_d1, c_d1]
                        diag1_elements.append(float(val_d1) if val_d1 != -1 else np.nan)
                if len(diag1_elements) > 0:
                    diag1_line_np = np.empty(len(diag1_elements), dtype=np.float64)
                    for i_fill in range(len(diag1_elements)): diag1_line_np[i_fill] = diag1_elements[i_fill]
                    sequences_found_d1 = BoardAnalyzerUtils.find_sequences_in_line_numba(
                        diag1_line_np, min_sequence_len_to_score, check_arithmetic, check_geometric, allow_gaps_in_sequence
                    )
                    for seq_idx_d1 in range(len(sequences_found_d1)):
                        seq_d1 = sequences_found_d1[seq_idx_d1]
                        is_val_in_seq_d1 = False
                        for item_in_seq_idx_d1 in range(len(seq_d1)):
                            if seq_d1[item_in_seq_idx_d1] == val_to_try : is_val_in_seq_d1 = True; break
                        if is_val_in_seq_d1:
                            seq_len_d1 = float(len(seq_d1)) #
                            if sequence_quality_weighting:
                                seq_sum_d1 = 0.0
                                for item_s_idx_d1 in range(len(seq_d1)): seq_sum_d1 += seq_d1[item_s_idx_d1]
                                avg_val_in_seq_d1 = seq_sum_d1 / len(seq_d1) if len(seq_d1) > 0 else 0.0 #
                                if max_board_val_f10 > 0 and avg_val_in_seq_d1 > (max_board_val_f10 * high_value_sequence_threshold_factor): #
                                    seq_len_d1 *= 1.2
                            current_val_max_len = max(current_val_max_len, seq_len_d1)
                
                # 4. Diagonal 2 (/)
                # For cell (r,c), elements on anti-diagonal have r+c = const
                diag2_sum_rc = r_idx + c_idx
                diag2_elements = numba.typed.List() # type: numba.typed.List[np.float_]
                for r_d2 in range(rows):
                    c_d2 = diag2_sum_rc - r_d2
                    if 0 <= c_d2 < cols : # Ensure c_d2 is valid
                        val_d2 = temp_grid[r_d2, c_d2]
                        diag2_elements.append(float(val_d2) if val_d2 != -1 else np.nan)
                if len(diag2_elements) > 0:
                    diag2_line_np = np.empty(len(diag2_elements), dtype=np.float64)
                    for i_fill_d2 in range(len(diag2_elements)): diag2_line_np[i_fill_d2] = diag2_elements[i_fill_d2]
                    sequences_found_d2 = BoardAnalyzerUtils.find_sequences_in_line_numba(
                        diag2_line_np, min_sequence_len_to_score, check_arithmetic, check_geometric, allow_gaps_in_sequence
                    )
                    for seq_idx_d2 in range(len(sequences_found_d2)):
                        seq_d2 = sequences_found_d2[seq_idx_d2]
                        is_val_in_seq_d2 = False
                        for item_in_seq_idx_d2 in range(len(seq_d2)):
                             if seq_d2[item_in_seq_idx_d2] == val_to_try : is_val_in_seq_d2 = True; break
                        if is_val_in_seq_d2:
                            seq_len_d2 = float(len(seq_d2)) #
                            if sequence_quality_weighting:
                                seq_sum_d2 = 0.0
                                for item_s_idx_d2 in range(len(seq_d2)): seq_sum_d2 += seq_d2[item_s_idx_d2]
                                avg_val_in_seq_d2 = seq_sum_d2 / len(seq_d2) if len(seq_d2) > 0 else 0.0 #
                                if max_board_val_f10 > 0 and avg_val_in_seq_d2 > (max_board_val_f10 * high_value_sequence_threshold_factor): #
                                    seq_len_d2 *= 1.2
                            current_val_max_len = max(current_val_max_len, seq_len_d2)

                if current_val_max_len >= min_sequence_len_to_score: #
                    max_len_contribution_for_this_cell = max(
                        max_len_contribution_for_this_cell, current_val_max_len
                    )
            
            if heuristic_max_len_val > 1e-9: # 來源：新大腦.pdf (Page 13)
                scores[r_idx, c_idx] = MathUtils.normalize_value( #
                    max_len_contribution_for_this_cell,
                    0.0, 
                    heuristic_max_len_val, #
                    clamp=True,
                )
            else: # 來源：新大腦.pdf (Page 14)
                scores[r_idx, c_idx] = 0.0
    return scores

def EXT_F10_Discontinuity_Vec(
    grid: np.ndarray,
    config: DiscontinuityRepairConfig, 
    request_id: str | None = "N/A_F10_Discontinuity", #
) -> np.ndarray:
    """
    (F10-不連續性修復/序列完成度) - Optimized with Numba
    核心規則:評估在空格填入數字後,是否能修復或完成某個方向上的數字序列(例如等差)。
    目的:偏好那些能夠「承先啟後」,使斷裂的序列得以延續或形成的空格。
    啟發式類型:序列與模式識別
    輸出詮釋:分數越高表示該空格填入某個合法數字後,能形成或延長的序列越長/越重要
    來源：新大腦.pdf - EXT_F10_Discontinuity_Vec (Page 12)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_F10"
    logger.debug(
        f"Executing EXT_F10_Discontinuity_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    if rows == 0 or cols == 0: #
        return np.zeros((rows,cols), dtype=float)

    legal_values_list = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 12)
    if not legal_values_list:
        return np.zeros_like(grid, dtype=float)
    legal_values_arr = np.array(legal_values_list, dtype=np.int_)


    min_seq_len_score = config.min_sequence_len_to_score #
    heuristic_max_len = float(max(rows, cols)) #
    if heuristic_max_len < min_seq_len_score: # 來源：新大腦.pdf (Page 12)
        heuristic_max_len = float(min_seq_len_score)
    if heuristic_max_len <= 1e-9: heuristic_max_len = 1.0 

    scores = EXT_F10_Discontinuity_Vec_numba(
        grid,
        legal_values_arr,
        min_seq_len_score,
        config.allow_gaps_in_sequence,
        config.check_arithmetic,
        config.check_geometric,
        config.sequence_quality_weighting,
        config.high_value_sequence_threshold_factor,
        heuristic_max_len
    )
    return scores * config.weight

# Helper for P7 Numba BFS
@njit
def _p7_bfs_numba(
    grid_p7: np.ndarray,
    r_start_p7: int, c_start_p7: int, #
    max_path_search_depth_p7: int,
    path_value_decay_factor_p7: float,
    target_value_min_threshold_p7: float,
    enable_target_value_filter: bool
) -> float:
    rows_p7, cols_p7 = grid_p7.shape
    current_placement_path_score: float = 0.0
    
    q_r = np.empty(rows_p7 * cols_p7, dtype=np.int_) # Max possible queue size
    q_c = np.empty(rows_p7 * cols_p7, dtype=np.int_)
    q_path_len = np.empty(rows_p7 * cols_p7, dtype=np.int_)
    
    q_head, q_tail = 0, 0

    q_r[q_tail] = r_start_p7
    q_c[q_tail] = c_start_p7
    q_path_len[q_tail] = 0
    q_tail += 1
    
    visited_for_bfs_p7 = np.zeros((rows_p7, cols_p7), dtype=np.bool_) #
    visited_for_bfs_p7[r_start_p7, c_start_p7] = True #
    
    head_count = 0 #
    max_bfs_steps_practical_p7 = (2 * max_path_search_depth_p7 + 1)**2 * 4 #

    while q_head < q_tail and head_count < max_bfs_steps_practical_p7: #
        head_count += 1
        curr_r, curr_c, path_len = q_r[q_head], q_c[q_head], q_path_len[q_head]
        q_head += 1

        # Explore neighbors (4-connectivity)
        # dr_dc_arr = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int_) # Not directly usable like this in Numba's prange
        for dr_p7_val, dc_p7_val in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 來源：新大腦.pdf (Page 15) #
            next_r, next_c = curr_r + dr_p7_val, curr_c + dc_p7_val

            if 0 <= next_r < rows_p7 and 0 <= next_c < cols_p7: #
                if grid_p7[next_r, next_c] != -1: #
                    reached_val = float(grid_p7[next_r, next_c]) #
                    if enable_target_value_filter and reached_val < target_value_min_threshold_p7: #
                        continue 

                    effective_path_len = path_len + 1 #
                    current_placement_path_score += reached_val / (effective_path_len**path_value_decay_factor_p7) #
                
                elif (not visited_for_bfs_p7[next_r, next_c]) and \
                     grid_p7[next_r, next_c] == -1 and \
                     path_len + 1 < max_path_search_depth_p7: # 來源：新大腦.pdf (Page 15) #
                    visited_for_bfs_p7[next_r, next_c] = True
                    if q_tail < q_r.shape[0]: # Check queue bounds
                        q_r[q_tail] = next_r
                        q_c[q_tail] = next_c
                        q_path_len[q_tail] = path_len + 1
                        q_tail += 1
                    # else: queue full, stop this path (should be rare with large enough queue)
    return current_placement_path_score


# 來源：新大腦.pdf - 5. EXT_P7_Pathfinding_Value_Vec (Page 14)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_P7強化建議
@njit(parallel=True)
def EXT_P7_Pathfinding_Value_Vec_numba(
    grid: np.ndarray,
    # legal_values_arr: np.ndarray, # Loop over val_to_try is kept for structure, but not used in PDF's path score
    max_path_search_depth: int, #
    path_value_decay_factor: float, #
    target_value_min_threshold: float,
    enable_target_filter: bool,
    heuristic_max_path_score_val: float # Renamed
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: #
        return scores

    for r_start in prange(rows): #
        for c_start in range(cols):
            if grid[r_start, c_start] != -1:
                continue
            
            # The PDF loop for val_to_try does not change the BFS result as written.
            # "The original grid is used to find *existing* numbers."
            # "The path itself can traverse other empty cells."
            # "BFS explores from (r_start, c_start) ... to existing numbers."
            # If val_to_try IS intended to be the starting value of the path (hypothetically placed),
            # the BFS or scoring would need to use it. The PDF's score formula
            # `reached_val / (effective_path_len ** ...)` uses `reached_val` (an *existing* number).
            # For "不可有任何簡化效能 只能增強", I make the BFS faster.
            # The score for (r_start, c_start) is computed once by BFS.
            
            path_score_for_cell = _p7_bfs_numba(
                grid, r_start, c_start,
                max_path_search_depth, path_value_decay_factor,
                target_value_min_threshold, enable_target_filter
            )
            
            if heuristic_max_path_score_val > 1e-9: #
                scores[r_start, c_start] = MathUtils.normalize_value( #
                    path_score_for_cell, 0.0, heuristic_max_path_score_val, clamp=True
                )
            else: #
                scores[r_start, c_start] = 0.0
    return scores

def EXT_P7_Pathfinding_Value_Vec(
    grid: np.ndarray,
    config: PathfindingValueConfig, #
    request_id: str | None = "N/A_P7_Pathfinding", #
) -> np.ndarray:
    """
    (P7-路徑尋找價值) - Optimized with Numba
    核心規則:評估在空格填入數字後,形成連接到其他現有數字的路徑的價值。
    目的:偏好那些能夠「橋接」盤面區域,或連接到高價值目標的空格。
    啟發式類型:連通性與圖論
    輸出詮釋:分數越高表示該空格填入某數字後,能形成更有價值的路徑(考慮路徑長度與連接到的數字大小)
    來源：新大腦.pdf - EXT_P7_Pathfinding_Value_Vec (Page 14)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_P7"
    logger.debug(
        f"Executing EXT_P7_Pathfinding_Value_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    if rows == 0 or cols == 0: return np.zeros((rows,cols), dtype=float) #

    # Legal values not directly used in the Numba core path score as per PDF, but kept for structure if needed
    # legal_values_list = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 14)
    # if not legal_values_list: # No legal moves means no p_val to loop, but BFS from empty cell is still possible
    #     return np.zeros_like(grid, dtype=float) * config.weight 
    # legal_values_arr = np.array(legal_values_list, dtype=np.int_)


    max_possible_val = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(rows, cols)) #
    if max_possible_val == 0: max_possible_val = 1.0

    heuristic_max_path_score = ( #
        (2 * config.max_path_search_depth + 1)**2 * max_possible_val / (1.0**config.path_value_decay_factor)
    )
    if heuristic_max_path_score <= 1e-9: heuristic_max_path_score = 1.0 #

    enable_target_filter_p7 = config.target_value_threshold_factor > 1e-9 #
    target_val_min_thresh_p7 = max_possible_val * config.target_value_threshold_factor

    scores = EXT_P7_Pathfinding_Value_Vec_numba(
        grid,
        config.max_path_search_depth,
        config.path_value_decay_factor,
        target_val_min_thresh_p7,
        enable_target_filter_p7,
        heuristic_max_path_score
    )
    return scores * config.weight


# 來源：新大腦.pdf - 6. EXT_R5_Resource_Control_Vec (Page 16)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_R5強化建議
# This module is largely arithmetic and can be Numba-jitted.
@njit(parallel=True)
def EXT_R5_Resource_Control_Vec_numba(
    grid: np.ndarray,
    # potential_numbers_to_place_arr: np.ndarray, # This is now passed as hypothetical_high_val_placed
    hypothetical_high_val_placed_r5: float,
    max_possible_val_on_grid_r5: float,
    w_row_completeness: float,
    w_col_completeness: float,
    w_value_capture: float
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores #

    for r_idx in prange(rows): #
        num_filled_in_row = 0
        for c_scan_row in range(cols):
            if grid[r_idx, c_scan_row] != -1:
                num_filled_in_row += 1
        
        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            # 1. Row Completeness
            row_completeness_score = (num_filled_in_row + 1.0) / cols if cols > 0 else 0.0 #

            # 2. Column Completeness
            num_filled_in_col = 0
            for r_scan_col in range(rows):
                if grid[r_scan_col, c_idx] != -1:
                    num_filled_in_col +=1
            col_completeness_score = (num_filled_in_col + 1.0) / rows if rows > 0 else 0.0 #
            
            # 3. Value Capture
            value_capture_score: float = 0.0 #
            if hypothetical_high_val_placed_r5 > 0 and max_possible_val_on_grid_r5 > 1e-9: #
                value_capture_score = MathUtils.normalize_value( #
                    hypothetical_high_val_placed_r5, 1.0, max_possible_val_on_grid_r5, clamp=True
                )
            
            total_weight = w_row_completeness + w_col_completeness + w_value_capture #
            if total_weight <= 1e-9: total_weight = 1.0 

            combined_score = ( #
                w_row_completeness * row_completeness_score +
                w_col_completeness * col_completeness_score +
                w_value_capture * value_capture_score
            ) / total_weight 

            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score, 0.0, 1.0, clamp=True) #
            
    return scores

def EXT_R5_Resource_Control_Vec(
    grid: np.ndarray,
    config: ResourceControlConfig, #
    request_id: str | None = "N/A_R5_ResourceCtrl", #
) -> np.ndarray:
    """
    (R5-資源控制) - Optimized with Numba
    核心規則:從資源控制角度評估填補位置的策略價值。資源可包括行/列的完成度、對高價值數字的獲取潜力等。
    目的:偏好那些能夠鞏固盤面控制權,或獲取潛在高價值數字的空格。
    啟發式類型:策略與控制
    輸出詮釋:分數越高表示該空格在填入數字後,對資源的控制(如行列完成度、高價值數字佔據)越強
    來源：新大腦.pdf - EXT_R5_Resource_Control_Vec (Page 16)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_R5" #
    logger.debug(
        f"Executing EXT_R5_Resource_Control_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    )

    rows, cols = grid.shape
    if rows == 0 or cols == 0: return np.zeros((rows,cols), dtype=float)

    potential_numbers_list = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 16)
    
    hypothetical_high_val: float = 0.0 #
    if potential_numbers_list:
        hypothetical_high_val = float(np.max(potential_numbers_list))

    max_possible_val = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(rows, cols)) #
    if max_possible_val == 0: max_possible_val = 1.0 #

    scores = EXT_R5_Resource_Control_Vec_numba(
        grid,
        hypothetical_high_val,
        max_possible_val,
        config.w_row_completeness,
        config.w_col_completeness,
        config.w_value_capture
    )
    return scores * config.weight


# 來源：新大腦.pdf - 7. EXT_GM1_Row_Control_Vec (Page 17)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM1強化建議
@njit(parallel=True)
def EXT_GM1_Row_Control_Vec_numba(
    grid: np.ndarray,
    avg_potential_num_to_place_gm1: float,
    max_val_board_gm1: float,
    use_advanced_sequence_detection: bool,
    min_len_for_sequence_score: int,
    allow_gaps_for_sequence_score: int,
    w_density: float,
    w_sum_score: float,
    w_sequence_score: float
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores #

    for r_idx in prange(rows): #
        current_row_values_sum_orig = 0.0
        num_filled_in_row_orig = 0
        for c_scan in range(cols):
            if grid[r_idx, c_scan] != -1:
                num_filled_in_row_orig +=1
                current_row_values_sum_orig += grid[r_idx, c_scan]

        for c_idx in range(cols):
            if grid[r_idx, c_idx] != -1:
                continue

            density_score = (num_filled_in_row_orig + 1.0) / cols if cols > 0 else 0.0 #

            potential_row_sum = current_row_values_sum_orig + avg_potential_num_to_place_gm1 #
            heuristic_max_row_sum = float(cols * max_val_board_gm1) #
            sum_score_val: float = 0.0 # Renamed
            if heuristic_max_row_sum > 1e-9: #
                sum_score_val = MathUtils.normalize_value( #
                    potential_row_sum, 0.0, heuristic_max_row_sum, clamp=True
                )

            seq_score_val: float = 0.0 # Renamed
            if use_advanced_sequence_detection:
                max_len_this_placement = 0.0
                # Create a temp copy of the row for sequence analysis
                temp_grid_row_slice_gm1 = grid[r_idx, :].copy().astype(np.float64) # Numba needs consistent types
                temp_grid_row_slice_gm1[c_idx] = avg_potential_num_to_place_gm1 # Use average
                
                # Replace -1 with np.nan for find_sequences_in_line_numba
                for k_gm1_nan in range(temp_grid_row_slice_gm1.shape[0]):
                    if grid[r_idx, k_gm1_nan] == -1 and k_gm1_nan != c_idx : # Original -1s become NaN
                         temp_grid_row_slice_gm1[k_gm1_nan] = np.nan
                
                sequences_gm1 = BoardAnalyzerUtils.find_sequences_in_line_numba( #
                    temp_grid_row_slice_gm1, 
                    min_len_for_sequence_score,
                    True, False, # Assuming check_arithmetic=True, check_geometric=False from typical usage
                    allow_gaps_for_sequence_score
                )
                for s_idx in range(len(sequences_gm1)):
                    s_gm1 = sequences_gm1[s_idx]
                    is_val_in_s_gm1 = False
                    for item_s_gm1_idx in range(len(s_gm1)):
                        if BoardAnalyzerUtils._is_close_numba(s_gm1[item_s_gm1_idx], avg_potential_num_to_place_gm1): # Compare float
                             is_val_in_s_gm1 = True; break
                    if is_val_in_s_gm1:
                       max_len_this_placement = max(max_len_this_placement, float(len(s_gm1)))
                if cols > 0: #
                    seq_score_val = MathUtils.normalize_value(max_len_this_placement, 0.0, float(cols), clamp=True) #
            else: # Original simplified logic
                # This simplified logic is harder to make fully Numba-compatible if potential_numbers_to_place (Set) is involved
                # For optimization, prioritizing the use_advanced_sequence_detection path.
                # Fallback for non-advanced:
                if 0 < c_idx < cols - 1: #
                    prev_val_gm1 = grid[r_idx, c_idx - 1]
                    next_val_gm1 = grid[r_idx, c_idx + 1]
                    if prev_val_gm1 != -1 and next_val_gm1 != -1: #
                        if (prev_val_gm1 + next_val_gm1) % 2 == 0: #
                            mend_val_gm1 = (prev_val_gm1 + next_val_gm1) // 2
                            # Check if mend_val is in potential_numbers (cannot do this directly in Numba without passing the list)
                            # Simplified: assume mend_val is valid for now if this path is taken
                            if abs(mend_val_gm1 - prev_val_gm1) > 1e-6 : #
                                seq_score_val = 0.75 
                elif (c_idx == 0 and cols > 1 and grid[r_idx, c_idx + 1] != -1 and \
                      abs(grid[r_idx, c_idx + 1] - avg_potential_num_to_place_gm1) > 1e-6) or \
                     (c_idx == cols - 1 and cols > 1 and grid[r_idx, c_idx - 1] != -1 and \
                      abs(avg_potential_num_to_place_gm1 - grid[r_idx, c_idx - 1]) > 1e-6): #
                      seq_score_val = 0.25 #

            total_weight_gm1 = w_density + w_sum_score + w_sequence_score #
            if total_weight_gm1 <= 1e-9: total_weight_gm1 = 1.0

            combined_score_gm1 = ( #
                w_density * density_score + w_sum_score * sum_score_val + w_sequence_score * seq_score_val
            ) / total_weight_gm1
            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score_gm1, 0.0, 1.0, clamp=True) #
            
    return scores

def EXT_GM1_Row_Control_Vec(
    grid: np.ndarray,
    config: LineControlConfig, #
    request_id: str | None = "N/A_GM1_RowCtrl", #
) -> np.ndarray:
    """
    (GM1-行控制力) - Optimized with Numba
    核心規則:評估在特定空格填入數字後,對該行的完成度、數值總和或序列形成的貢獻。
    目的:偏好那些能增強單行控制力或形成有價值行模式的填補。
    啟發式類型:線性結構控制(行)
    輸出詮釋:分數越高表示對該行的潛在控制力或完成度越強
    來源：新大腦.pdf - EXT_GM1_Row_Control_Vec (Page 17)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM1" #
    logger.debug(
        f"Executing EXT_GM1_Row_Control_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    ) #

    rows, cols = grid.shape
    if rows == 0 or cols == 0: return np.zeros((rows,cols), dtype=float)

    potential_numbers_list_gm1 = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 18)
    avg_potential_num: float = 0.0 #
    if potential_numbers_list_gm1:
        avg_potential_num = float(np.mean(potential_numbers_list_gm1))

    max_val_b = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(rows, cols)) #
    if max_val_b == 0: max_val_b = 1.0

    scores = EXT_GM1_Row_Control_Vec_numba(
        grid,
        avg_potential_num,
        max_val_b,
        config.use_advanced_sequence_detection,
        config.min_len_for_sequence_score,
        config.allow_gaps_for_sequence_score,
        config.w_density,
        config.w_sum_score,
        config.w_sequence_score
    )
    return scores * config.weight


# 來源：新大腦.pdf - 8. EXT_GM2_Col_Flow_Vec (Page 19)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM2強化建議
# This function is symmetric to GM1, so Numba optimization is similar.
@njit(parallel=True)
def EXT_GM2_Col_Flow_Vec_numba(
    grid: np.ndarray,
    avg_potential_num_to_place_gm2: float,
    max_val_board_gm2: float,
    use_advanced_sequence_detection: bool,
    min_len_for_sequence_score: int,
    allow_gaps_for_sequence_score: int,
    w_density: float,
    w_sum_score: float,
    w_sequence_score: float
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores #

    for c_idx in prange(cols): #
        current_col_values_sum_orig = 0.0
        num_filled_in_col_orig = 0
        for r_scan in range(rows):
            if grid[r_scan, c_idx] != -1: #
                num_filled_in_col_orig +=1
                current_col_values_sum_orig += grid[r_scan, c_idx]

        for r_idx in range(rows):
            if grid[r_idx, c_idx] != -1:
                continue

            density_score = (num_filled_in_col_orig + 1.0) / rows if rows > 0 else 0.0 #

            potential_col_sum = current_col_values_sum_orig + avg_potential_num_to_place_gm2 #
            heuristic_max_col_sum = float(rows * max_val_board_gm2) #
            sum_score_val: float = 0.0 #
            if heuristic_max_col_sum > 1e-9: #
                sum_score_val = MathUtils.normalize_value( #
                    potential_col_sum, 0.0, heuristic_max_col_sum, clamp=True
                )

            seq_score_val: float = 0.0 #
            if use_advanced_sequence_detection:
                max_len_this_placement = 0.0
                temp_grid_col_slice_gm2 = grid[:, c_idx].copy().astype(np.float64) #
                temp_grid_col_slice_gm2[r_idx] = avg_potential_num_to_place_gm2 #
                
                for k_gm2_nan in range(temp_grid_col_slice_gm2.shape[0]):
                    if grid[k_gm2_nan, c_idx] == -1 and k_gm2_nan != r_idx :
                         temp_grid_col_slice_gm2[k_gm2_nan] = np.nan

                sequences_gm2 = BoardAnalyzerUtils.find_sequences_in_line_numba( #
                    temp_grid_col_slice_gm2,
                    min_len_for_sequence_score,
                    True, False, # Assuming Arithmetic=T, Geometric=F
                    allow_gaps_for_sequence_score
                )
                for s_idx_c in range(len(sequences_gm2)):
                    s_gm2 = sequences_gm2[s_idx_c]
                    is_val_in_s_gm2 = False
                    for item_s_gm2_idx in range(len(s_gm2)):
                         if BoardAnalyzerUtils._is_close_numba(s_gm2[item_s_gm2_idx], avg_potential_num_to_place_gm2):
                            is_val_in_s_gm2 = True; break
                    if is_val_in_s_gm2:
                       max_len_this_placement = max(max_len_this_placement, float(len(s_gm2)))
                if rows > 0: #
                    seq_score_val = MathUtils.normalize_value(max_len_this_placement, 0.0, float(rows), clamp=True) #
            else: # Simplified logic
                if 0 < r_idx < rows - 1: #
                    prev_val_gm2 = grid[r_idx - 1, c_idx]
                    next_val_gm2 = grid[r_idx + 1, c_idx]
                    if prev_val_gm2 != -1 and next_val_gm2 != -1: #
                        if (prev_val_gm2 + next_val_gm2) % 2 == 0: #
                            mend_val_gm2 = (prev_val_gm2 + next_val_gm2) // 2
                            if abs(mend_val_gm2 - prev_val_gm2) > 1e-6: #
                                seq_score_val = 0.75
                elif (r_idx == 0 and rows > 1 and grid[r_idx + 1, c_idx] != -1 and \
                      abs(grid[r_idx + 1, c_idx] - avg_potential_num_to_place_gm2) > 1e-6) or \
                     (r_idx == rows - 1 and rows > 1 and grid[r_idx - 1, c_idx] != -1 and \
                      abs(avg_potential_num_to_place_gm2 - grid[r_idx - 1, c_idx]) > 1e-6): #
                      seq_score_val = 0.25 #

            total_weight_gm2 = w_density + w_sum_score + w_sequence_score #
            if total_weight_gm2 <= 1e-9: total_weight_gm2 = 1.0

            combined_score_gm2 = ( #
                w_density * density_score + w_sum_score * sum_score_val + w_sequence_score * seq_score_val
            ) / total_weight_gm2
            scores[r_idx, c_idx] = MathUtils.normalize_value(combined_score_gm2, 0.0, 1.0, clamp=True) #

    return scores

def EXT_GM2_Col_Flow_Vec(
    grid: np.ndarray,
    config: LineControlConfig, # Reuses LineControlConfig #
    request_id: str | None = "N/A_GM2_ColCtrl", #
) -> np.ndarray:
    """
    (GM2 - 列流動性/列控制力) - Optimized with Numba
    核心規則:評估在特定空格填入數字後,對該列的完成度、數值總和或序列形成的貢獻。
    目的:偏好那些能增強單列控制力或形成有價值列模式的填補。
    啟發式類型:線性結構控制(列)
    輸出詮釋:分數越高表示對該列的潛在控制力或完成度越強
    來源：新大腦.pdf - EXT_GM2_Col_Flow_Vec (Page 19-20)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM2" #
    logger.debug(
        f"Executing EXT_GM2_Col_Flow_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    ) #

    rows, cols = grid.shape
    if rows == 0 or cols == 0: return np.zeros((rows,cols), dtype=float)

    potential_numbers_list_gm2 = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 20)
    avg_potential_num_gm2: float = 0.0 #
    if potential_numbers_list_gm2:
        avg_potential_num_gm2 = float(np.mean(potential_numbers_list_gm2))

    max_val_b_gm2 = float(BoardAnalyzerUtils.get_card_max_value_from_grid_dimensions(rows, cols)) #
    if max_val_b_gm2 == 0: max_val_b_gm2 = 1.0

    scores = EXT_GM2_Col_Flow_Vec_numba(
        grid,
        avg_potential_num_gm2,
        max_val_b_gm2,
        config.use_advanced_sequence_detection,
        config.min_len_for_sequence_score,
        config.allow_gaps_for_sequence_score,
        config.w_density,
        config.w_sum_score,
        config.w_sequence_score
    )
    return scores * config.weight


# 來源：新大腦.pdf - 9. EXT_GM3_Adv_Connected_Comp_Vec (Page 21)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM3強化建議
# Numba can optimize the BFS part of this.
@njit
def _gm3_bfs_component_analysis(
    grid_gm3: np.ndarray, 
    r_start_gm3: int, c_start_gm3: int,
    visited_overall_gm3: np.ndarray # bool array, modified by this function
) -> Tuple[float, float, float]: # area_size, compactness, avg_value (avg_value not used by PDF score for GM3 empty cells)
    rows_gm3, cols_gm3 = grid_gm3.shape
    component_cells_coords_r = numba.typed.List() # type: numba.typed.List[np.int_]
    component_cells_coords_c = numba.typed.List() # type: numba.typed.List[np.int_]
    
    q_r_gm3 = np.empty(rows_gm3 * cols_gm3, dtype=np.int_)
    q_c_gm3 = np.empty(rows_gm3 * cols_gm3, dtype=np.int_)
    q_head_gm3, q_tail_gm3 = 0, 0

    q_r_gm3[q_tail_gm3] = r_start_gm3
    q_c_gm3[q_tail_gm3] = c_start_gm3
    q_tail_gm3 += 1
    
    visited_overall_gm3[r_start_gm3, c_start_gm3] = True # Mark as globally visited #

    min_r_bbox, max_r_bbox = r_start_gm3, r_start_gm3
    min_c_bbox, max_c_bbox = c_start_gm3, c_start_gm3

    while q_head_gm3 < q_tail_gm3: #
        r_curr, c_curr = q_r_gm3[q_head_gm3], q_c_gm3[q_head_gm3]
        q_head_gm3 += 1
        
        component_cells_coords_r.append(r_curr)
        component_cells_coords_c.append(c_curr)

        min_r_bbox = min(min_r_bbox, r_curr)
        max_r_bbox = max(max_r_bbox, r_curr)
        min_c_bbox = min(min_c_bbox, c_curr)
        max_c_bbox = max(max_c_bbox, c_curr)

        for dr_gm3, dc_gm3 in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #
            nr, nc = r_curr + dr_gm3, c_curr + dc_gm3

            if 0 <= nr < rows_gm3 and 0 <= nc < cols_gm3 and \
               grid_gm3[nr, nc] == -1 and not visited_overall_gm3[nr, nc]: # Check for empty and not visited #
                visited_overall_gm3[nr, nc] = True
                if q_tail_gm3 < q_r_gm3.shape[0]:
                    q_r_gm3[q_tail_gm3] = nr
                    q_c_gm3[q_tail_gm3] = nc
                    q_tail_gm3 += 1
            
    area_size_res = float(len(component_cells_coords_r)) #
    compactness_res: float = 0.0
    if area_size_res > 0:
        bbox_height = float(max_r_bbox - min_r_bbox + 1) #
        bbox_width = float(max_c_bbox - min_c_bbox + 1) #
        bbox_area = bbox_height * bbox_width #
        if bbox_area > 1e-9: #
            compactness_res = area_size_res / bbox_area
    
    # For GM3, which scores empty cells based on the area of the empty component they belong to,
    # avg_value of the component is not directly used in the PDF's score for that empty cell.
    # So, returning 0.0 for avg_value here.
    return area_size_res, compactness_res, 0.0, component_cells_coords_r, component_cells_coords_c


@njit(parallel=False) # Parallelizing the outer loop over cells might be complex due to shared visited_overall
def EXT_GM3_Adv_Connected_Comp_Vec_numba(
    grid: np.ndarray,
    consider_shape_factor: bool,
    shape_factor_weight: float
) -> np.ndarray:
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=np.float64)
    if rows == 0 or cols == 0: return scores #

    visited_overall_gm3_main = np.zeros((rows, cols), dtype=np.bool_) #

    for r_start_main in range(rows): #
        for c_start_main in range(cols):
            if visited_overall_gm3_main[r_start_main, c_start_main] or grid[r_start_main, c_start_main] != -1: #
                continue

            area_size_comp, compactness_comp, _, comp_r_coords, comp_c_coords = _gm3_bfs_component_analysis(
                grid, r_start_main, c_start_main, visited_overall_gm3_main
            )
            
            total_cells = float(rows * cols) #
            norm_area_size: float = 0.0 #
            if total_cells > 1e-9: #
                norm_area_size = MathUtils.normalize_value(area_size_comp, 0.0, total_cells, clamp=True) #
            
            final_component_score = norm_area_size
            if consider_shape_factor: #
                norm_compactness = MathUtils.normalize_value(compactness_comp, 0.0, 1.0, clamp=True) # Compactness is already 0-1
                final_component_score = (1.0 - shape_factor_weight) * norm_area_size + \
                                        shape_factor_weight * norm_compactness #
                final_component_score = MathUtils.normalize_value(final_component_score, 0.0, 1.0, clamp=True) #

            for i_cell in range(len(comp_r_coords)): #
                scores[comp_r_coords[i_cell], comp_c_coords[i_cell]] = final_component_score
                
    return scores

def EXT_GM3_Adv_Connected_Comp_Vec(
    grid: np.ndarray,
    config: ConnectedComponentConfig, #
    request_id: str | None = "N/A_GM3_ConnComp", #
) -> np.ndarray:
    """
    (GM3 - 高級連通元件分析-空格區域) - Optimized with Numba
    核心規則:分析空格所屬的連續空格區域的大小。
    目的:偏好那些屬於較大連續空格區域的空格,這些區域可能提供更大的填補潛力或形成大型結構的機會。
    啟發式類型:連通元件分析(針對空格)
    輸出詮釋:分數越高表示該空格屬於一個面積越大的連續空格區域(分數經盤面總大小正規化)
    來源：新大腦.pdf - EXT_GM3_Adv_Connected_Comp_Vec (Page 21)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else "N/A_brain_GM3" #
    logger.debug(
        f"Executing EXT_GM3_Adv_Connected_Comp_Vec with config: {config.model_dump_json(indent=2)}",
        extra={"request_id": effective_request_id},
    ) #

    scores = EXT_GM3_Adv_Connected_Comp_Vec_numba(
        grid,
        config.consider_shape_factor,
        config.shape_factor_weight
    )
    return scores * config.weight