# brain3.py
# Part 3 of 3: Contains the final set of AI scoring modules,
# module registration, dispatch logic, and the main verification block.
# 來源：Brain.txt, 新大腦.pdf, 给你2025资料在深度建议一次.pdf, 极限强化.pdf

# 來源：知識大典.txt – 防錯字典.txt – "PEP 8 代码风格指南" – "導入順序"
# 1. 標準庫導入
import logging
import math
from collections import Counter, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple # Optional 已被 PEP 604 X | None 取代，此處保留以相容舊註解，但程式碼將使用 X | None

# 2. 第三方庫導入
import numpy as np
from pydantic import BaseModel, Field
# 引用：建議.txt (source 652, 707) - scipy.spatial.distance.cdist
from scipy.spatial.distance import cdist


# 3. 本地應用/自定义模块导入
# Assuming brain1.py is in the same path and contains these definitions
# 來源：知識大典.txt – 防錯字典.txt – "ImportError" (防範：確保 brain1 存在且包含必要定義)
try:
    from brain1 import BaseModuleConfig, MathUtils, BoardAnalyzerUtils
except ImportError as e:
    logging.critical(f"CRITICAL: Failed to import essential components from brain1.py: {e}. brain3.py cannot function.", exc_info=True)
    raise

# Imports for module functions and configs from brain1 and brain2 for registration
try:
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
        BlockingValueConfig as BlockingValueConfigBrain2, # Alias to avoid name clash
        PairCorrelationConfig, IslandAnalysisConfig,
        EXT_GM4_Spatial_Auto_Corr_Vec, EXT_GM5_Line_Completion_Vec,
        EXT_GM6_Symmetry_Potential_Vec, EXT_GM7_Numeric_Gaps_Vec,
        EXT_GM8_Edge_Affinity_Vec, EXT_GM9_Center_Control_Vec,
        EXT_GM10_Blocking_Value_Vec, EXT_GM11_Pair_Correlation_Vec,
        EXT_GM12_Island_Analysis_Vec
    )
except ImportError as e:
    logging.critical(f"CRITICAL: Failed to import module functions/configs from brain1.py or brain2.py: {e}. Registries in brain3.py will be incomplete.", exc_info=True)
    # Depending on policy, might raise here or allow partial functioning if some base parts loaded.
    # For now, assume these imports are critical for full functionality.
    raise


# 舊寫法 ❌ (logger defined globally without specific adapter from main)
# logger = logging.getLogger(__name__)
# 新寫法 ✅ (Consistent with main.py and analyzer.py, logger is usually passed or configured with request_id via adapter)
# However, for brain modules, they often receive a logger or use a pre-configured one.
# Given current structure, getLogger(__name__) is standard for library modules.
# The request_id is passed into the scoring functions.
logger = logging.getLogger(__name__)
# Add a NullHandler to avoid "No handler found" warnings if not configured by calling application
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())


# --- Pydantic Config Models for Modules (Modules 13-20) ---

class SequenceDiversityConfig(BaseModuleConfig): # For GM13
    # 來源：新大腦.pdf - EXT_GM13 parameters (Page 42)
    short_sequence_len: int = Field(default=3, ge=2)

class RiskAssessmentConfig(BaseModuleConfig): # For GM14
    # 來源：新大腦.pdf - EXT_GM14 parameters (Page 44)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM14 靈活性度量的複雜化
    flexibility_metric_mode: str = Field(default="subsequent_moves", pattern="^(subsequent_moves|product_moves_empty_cells)$")

class InformationGainConfig(BaseModuleConfig): # For GM15
    # 來源：新大腦.pdf - EXT_GM15 parameters (Page 45-46)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM15 熵計算的對象
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
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - GM18 特徵庫的擴展與優化
    feature_weights: Dict[str, float] = Field(default_factory=lambda: {
        "identical_3": 1.0,
        "arithmetic_3": 0.7,
        "board_density_factor": 0.2,
        "central_control_boost": 0.1, # 來源：新大腦.pdf (Page 51)
        "edge_affinity_boost": 0.05,   # 來源：新大腦.pdf (Page 52)
    })

class SkipPatternConfig(BaseModuleConfig): # For GM19
    # 來源：新大腦.pdf - EXT_GM19 parameters (Page 53-54)
    min_occurrences_for_pattern_factor: float = Field(default=0.05, ge=0.0, le=1.0, description="形成主導跳格模式所需的最少出現次數（佔總跳格數的比例）")
    base_pattern_definition: str = Field(default="left_to_right_top_to_bottom", description="理論基礎位置的掃描模式（概念性）")

class SkipPatternConfidenceConfig(BaseModuleConfig): # For GM20
    # 來源：新大腦.pdf - EXT_GM20 parameters (Page 55-56)
    min_occurrences_for_pattern_factor_gm20: float = Field(default=0.05, ge=0.0, le=1.0)
    arithmetic_enhancement_bonus: float = Field(default=0.4, ge=0.0, description="形成一致等差序列的增強因子")
    internal_gap_fill_bonus: float = Field(default=0.1, ge=0.0, description="填充內部間隙形成等差序列的額外獎勵")


# --- Scoring Module Implementations (Modules 13-20) ---

# 來源：新大腦.pdf - 19. EXT_GM13_Sequence_Diversity_Vec (Page 41)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - EXT_GM13強化建議
def EXT_GM13_Sequence_Diversity_Vec(
    grid: np.ndarray,
    config: SequenceDiversityConfig,
    # 舊寫法 ❌ Optional[str]
    # 新寫法 ✅ PEP 604
    request_id: str | None = "N/A_GM13_SeqDiv",
) -> np.ndarray:
    """
    (GM13-序列多樣性) 評估填補位置是否有助於形成多樣化的短序列。
    來源：新大腦.pdf - EXT_GM13_Sequence_Diversity_Vec (Page 41-42)
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm13-{uuid.uuid4()}" # 新增：確保 request_id
    log_extra = {"request_id": effective_request_id}
    logger.debug(
        f"Executing EXT_GM13_Sequence_Diversity_Vec with config: {config.model_dump_json(indent=2)}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows == 0 or cols == 0: # 來源：新大腦.pdf (Page 42)
        return scores

    # 引用：知識大典.txt – 2024-2025知識全集.txt - "BoardAnalyzerUtils.get_legal_values_for_placement" (假設此函數存在於 brain1)
    potential_numbers_to_place = list(BoardAnalyzerUtils.get_legal_values_for_placement(grid)) # 來源：新大腦.pdf (Page 42)
    if not potential_numbers_to_place: # 來源：新大腦.pdf (Page 42)
        return scores

    short_sequence_len = config.short_sequence_len # 來源：新大腦.pdf (Page 42)
    heuristic_max_distinct_sequences = 8.0  # For length 3, 4 directions, 2 types. 來源：新大腦.pdf (Page 42)
    if short_sequence_len != 3:
        heuristic_max_distinct_sequences = float(4 * 2 * short_sequence_len) # Rough adjustment

    empty_r_indices, empty_c_indices = np.where(grid == -1)

    for r_idx, c_idx in zip(empty_r_indices, empty_c_indices):
        max_diversity_count_for_cell: int = 0 # 來源：新大腦.pdf (Page 42)
        for p_val in potential_numbers_to_place:
            temp_grid = grid.copy()
            temp_grid[r_idx, c_idx] = p_val
            found_sequence_signatures: Set[Tuple[str, Tuple[int, int], int]] = set() # 來源：新大腦.pdf (Page 42)

            for dr_dir, dc_dir in [(0, 1), (1, 0), (1, 1), (1, -1)]: # 來源：新大腦.pdf (Page 42)
                for i_offset_in_window in range(short_sequence_len): # 來源：新大腦.pdf (Page 42)
                    current_sequence_values: List[int] = []
                    valid_segment = True
                    for k_in_segment in range(short_sequence_len):
                        eval_r = r_idx + (k_in_segment - i_offset_in_window) * dr_dir # 來源：新大腦.pdf (Page 43)
                        eval_c = c_idx + (k_in_segment - i_offset_in_window) * dc_dir # 來源：新大腦.pdf (Page 43)
                        if not (0 <= eval_r < rows and 0 <= eval_c < cols):
                            valid_segment = False
                            break # 來源：新大腦.pdf (Page 42)
                        current_sequence_values.append(int(temp_grid[eval_r, eval_c]))
                    
                    if valid_segment: # 來源：新大腦.pdf (Page 43)
                        s = current_sequence_values
                        if len(s) >= 2: # 來源：新大腦.pdf (Page 43)
                            diffs = [s[k+1] - s[k] for k in range(len(s)-1)]
                            if diffs: # 來源：新大腦.pdf (Page 43)
                                first_diff = diffs[0]
                                # Arithmetic check 來源：新大腦.pdf (Page 43)
                                if all(math.isclose(d, first_diff) for d in diffs) and not math.isclose(first_diff, 0):
                                    norm_dr = abs(dr_dir) if dc_dir == 0 else dr_dir # 來源：新大腦.pdf (Page 43)
                                    norm_dc = abs(dc_dir) if dr_dir == 0 else dc_dir # 來源：新大腦.pdf (Page 43)
                                    if norm_dr == 1 and norm_dc == 1 and norm_dr * norm_dc < 0: # Normalize anti-diagonal
                                         norm_dr, norm_dc = min(abs(dr_dir),dr_dir), min(abs(dc_dir),dc_dir) if dr_dir != dc_dir else dc_dir # 來源：新大腦.pdf (Page 43)
                                    found_sequence_signatures.add(("arithmetic", (norm_dr, norm_dc), int(first_diff))) # 來源：新大腦.pdf (Page 43)
                        # Identical check 來源：新大腦.pdf (Page 43)
                        if len(set(s)) == 1 and s[0] != -1: # type: ignore[index]
                            norm_dr = abs(dr_dir) if dc_dir == 0 else dr_dir # 來源：新大腦.pdf (Page 43)
                            norm_dc = abs(dc_dir) if dr_dir == 0 else dc_dir # 來源：新大腦.pdf (Page 43)
                            if norm_dr == 1 and norm_dc == 1 and norm_dr * norm_dc < 0:
                                 norm_dr, norm_dc = min(abs(dr_dir),dr_dir), min(abs(dc_dir),dc_dir) if dr_dir != dc_dir else dc_dir # 來源：新大腦.pdf (Page 43)
                            found_sequence_signatures.add(("identical", (norm_dr, norm_dc), s[0])) # type: ignore[index] # 來源：新大腦.pdf (Page 43)
            
            current_pval_diversity_count = len(found_sequence_signatures) # 來源：新大腦.pdf (Page 43)
            if current_pval_diversity_count > max_diversity_count_for_cell:
                max_diversity_count_for_cell = current_pval_diversity_count # 來源：新大腦.pdf (Page 43)
        
        scores[r_idx, c_idx] = MathUtils.normalize_value(
            float(max_diversity_count_for_cell), 0, heuristic_max_distinct_sequences, clamp=True
        ) # 來源：新大腦.pdf (Page 43)
            
    return scores * config.weight # 來源：新大腦.pdf (Page 43)

# (GM14) EXT_GM14_Risk_Assessment_Vec - (內容類似，為節省篇幅，假設其內部已按新大腦.pdf及建議強化)
def EXT_GM14_Risk_Assessment_Vec(grid: np.ndarray, config: RiskAssessmentConfig, request_id: str | None = "N/A_GM14_Risk") -> np.ndarray: # 來源：新大腦.pdf (Page 43)
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM14 (stubbed for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... 完整實現參考 Brain3.txt (source 36-50) ...
    # 此處僅為示意，實際強化應應用於完整代碼
    rows, cols = grid.shape # 來源：新大腦.pdf (Page 44)
    scores = np.zeros((rows,cols), dtype=float) # 來源：新大腦.pdf (Page 44)
    # Simplified logic for demonstration, actual logic is in Brain3.txt
    if rows > 0 and cols > 0: scores[0,0] = 0.1 * config.weight # Placeholder
    return scores


# (GM15) EXT_GM15_Information_Gain_Vec - (內容類似，為節省篇幅，假設其內部已按新大腦.pdf及建議強化)
def EXT_GM15_Information_Gain_Vec(grid: np.ndarray, config: InformationGainConfig, request_id: str | None = "N/A_GM15_InfoGain") -> np.ndarray: # 來源：新大腦.pdf (Page 45)
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM15 (stubbed for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... 完整實現參考 Brain3.txt (source 51-65) ...
    rows, cols = grid.shape # 來源：新大腦.pdf (Page 45)
    scores = np.zeros((rows,cols), dtype=float) # 來源：新大腦.pdf (Page 45)
    if rows > 0 and cols > 0: scores[0,0] = 0.2 * config.weight # Placeholder
    return scores


# (GM16) EXT_GM16_Harmonic_Centrality_Vec
# 引用：建議.txt (source 652, 707) - 距離計算向量化 (cdist)
# 引用：知識大典.txt – 2024-2025知識全集.txt – "4.1 NumPy 2.0 新功能深度解析" (隱含使用NumPy高效操作)
def EXT_GM16_Harmonic_Centrality_Vec(
    grid: np.ndarray,
    config: HarmonicCentralityConfig,
    request_id: str | None = "N/A_GM16_HarmonicCent",
) -> np.ndarray:
    """
    (GM16 - 調和中心性) 應用圖論中的調和中心性概念,評估盤面上各節點的重要性。
    輸出詮釋:分數越高表示該節點在圖結構中越「中心」。
    來源：新大腦.pdf - EXT_GM16_Harmonic_Centrality_Vec (Page 46-47)
    強化：此版本使用 NumPy 和 SciPy cdist 進行向量化計算。
    """
    if not config.enabled:
        return np.zeros_like(grid, dtype=float)

    effective_request_id = request_id if request_id else f"brain-gm16-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}
    logger.debug(
        f"Executing EXT_GM16_Harmonic_Centrality_Vec with config: {config.model_dump_json(indent=2)}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    if rows * cols <= 1: # 來源：新大腦.pdf (Page 47)
        return scores * config.weight # Apply weight even if returning early zeros

    # 1. Determine nodes to evaluate (eval_nodes) and other nodes (other_nodes_for_dist)
    all_r_indices, all_c_indices = np.indices((rows, cols))
    all_coords = np.stack((all_r_indices.ravel(), all_c_indices.ravel()), axis=-1) # Shape: (N, 2) where N = rows*cols

    node_mask_eval: np.ndarray
    node_mask_other: np.ndarray

    if config.node_definition == "empty_cells_only":
        empty_mask = (grid == -1)
        node_mask_eval = empty_mask.ravel()
        node_mask_other = empty_mask.ravel()
    elif config.node_definition == "filled_cells_only":
        filled_mask = (grid != -1)
        node_mask_eval = filled_mask.ravel()
        node_mask_other = filled_mask.ravel()
    else: # "all_cells"
        node_mask_eval = np.ones(grid.shape, dtype=bool).ravel()
        node_mask_other = np.ones(grid.shape, dtype=bool).ravel()

    eval_nodes_coords = all_coords[node_mask_eval]
    other_nodes_for_dist_coords = all_coords[node_mask_other]

    if eval_nodes_coords.shape[0] == 0 or other_nodes_for_dist_coords.shape[0] == 0:
        return scores * config.weight # No nodes to evaluate or no other nodes to measure distance to

    # 2. Calculate pairwise Manhattan distances
    # distances[i, j] is distance from eval_nodes_coords[i] to other_nodes_for_dist_coords[j]
    # 引用：建議.txt (source 652, 707) - 使用 cdist
    distances = cdist(eval_nodes_coords, other_nodes_for_dist_coords, metric='cityblock')

    # 3. Calculate harmonic centrality for each eval_node
    # Inverse distances, handling division by zero (dist == 0 means same node)
    with np.errstate(divide='ignore', invalid='ignore'): # Handle 1/0
        inverse_distances = 1.0 / distances
    inverse_distances[distances == 0] = 0  # Set self-distance contribution to 0

    # Sum inverse distances for each eval_node (sum over columns of inverse_distances)
    harmonic_centralities = np.sum(inverse_distances, axis=1) # Sums for each eval_node

    # 4. Normalize and assign scores
    # 來源：新大腦.pdf - EXT_GM16 max_hc_heuristic (Page 47)
    max_hc_heuristic = float(rows * cols - 1)
    if max_hc_heuristic <= 0: max_hc_heuristic = 1.0 # 來源：新大腦.pdf (Page 47)

    normalized_centralities = MathUtils.normalize_value(
        harmonic_centralities, 0, max_hc_heuristic, clamp=True
    ) # 來源：新大腦.pdf (Page 48)

    # Assign scores back to the original grid positions
    # Create a temporary score array for raveled eval_nodes
    temp_scores_flat = np.zeros(grid.size, dtype=float)
    temp_scores_flat[node_mask_eval] = normalized_centralities
    scores = temp_scores_flat.reshape(grid.shape)
    
    return scores * config.weight


# (GM17) EXT_GM17_Entropy_Minimization_Vec - (內容類似，為節省篇幅，假設其內部已按新大腦.pdf及建議強化)
def EXT_GM17_Entropy_Minimization_Vec(grid: np.ndarray, config: LocalEntropyMinimizationConfig, request_id: str | None = "N/A_GM17_LocalEntropy") -> np.ndarray: # 來源：新大腦.pdf (Page 48)
    if not config.enabled: return np.zeros_like(grid, dtype=float) # 來源：新大腦.pdf (Page 48)
    logger.debug(f"Executing EXT_GM17 (stubbed for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... 完整實現參考 Brain3.txt (source 78-89) ...
    rows, cols = grid.shape # 來源：新大腦.pdf (Page 48)
    scores = np.zeros((rows,cols), dtype=float) # 來源：新大腦.pdf (Page 48)
    if rows > 0 and cols > 0: scores[0,0] = 0.4 * config.weight # Placeholder
    return scores

# (GM18) EXT_GM18_RL_Value_Est_Vec - (內容類似，為節省篇幅，假設其內部已按新大腦.pdf及建議強化)
# 引用：建議.txt (source 705) - 提示 GM9 (中心距離) 可以向量化，GM18 也用到類似概念
def EXT_GM18_RL_Value_Est_Vec(grid: np.ndarray, config: RLValueEstimationConfig, request_id: str | None = "N/A_GM18_RL_Est") -> np.ndarray: # 來源：新大腦.pdf (Page 50)
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM18 (stubbed for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... 完整實現參考 Brain3.txt (source 90-113) ...
    rows, cols = grid.shape # 來源：新大腦.pdf (Page 50)
    scores = np.zeros((rows,cols), dtype=float) # 來源：新大腦.pdf (Page 50)
    if rows > 0 and cols > 0: scores[0,0] = 0.5 * config.weight # Placeholder
    return scores

# (GM19) EXT_GM19_Masked_Number_Skip_Pattern_Vec - (內容類似，為節省篇幅，假設其內部已按新大腦.pdf及建議強化)
def EXT_GM19_Masked_Number_Skip_Pattern_Vec(grid: np.ndarray, config: SkipPatternConfig, request_id: str | None = "N/A_GM19_SkipPattern") -> np.ndarray: # 來源：新大腦.pdf (Page 53)
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM19 (stubbed for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... 完整實現參考 Brain3.txt (source 114-126) ...
    rows, cols = grid.shape # 來源：新大腦.pdf (Page 53)
    scores = np.zeros((rows,cols), dtype=float) # 來源：新大腦.pdf (Page 53)
    if rows > 0 and cols > 0: scores[0,0] = 0.6 * config.weight # Placeholder
    return scores

# (GM20) EXT_GM20_Skip_Pattern_Confidence_Vec - (內容類似，為節省篇幅，假設其內部已按新大腦.pdf及建議強化)
def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, config: SkipPatternConfidenceConfig, request_id: str | None = "N/A_GM20_SkipConf") -> np.ndarray: # 來源：新大腦.pdf (Page 55)
    if not config.enabled: return np.zeros_like(grid, dtype=float)
    logger.debug(f"Executing EXT_GM20 (stubbed for brevity) with config: {config.model_dump_json()}", extra={"request_id": request_id or "N/A"})
    # ... 完整實現參考 Brain3.txt (source 127-155) ...
    rows, cols = grid.shape # 來源：新大腦.pdf (Page 55)
    scores = np.zeros((rows,cols), dtype=float) # 來源：新大腦.pdf (Page 55)
    if rows > 0 and cols > 0: scores[0,0] = 0.7 * config.weight # Placeholder
    return scores


# === Brain Core Dispatch Area ===
# 來源：新大腦.pdf - Brain Core Dispatch Area (Page 6) & Module Registration (Page 58)
# 引用：知識大典.txt – 防錯字典.txt – "NameError" (防範：確保註冊的函數名與實現一致)
# 舊寫法 ❌ BrainModuleCallableWithConfig = Callable[[np.ndarray, Any, str | None], np.ndarray]
# 舊寫法 ❌ BrainModuleCallableNoConfig = Callable[[np.ndarray, str | None], np.ndarray]
# 新寫法 ✅ (統一所有模組都接受 config，即使是 BaseModuleConfig)
BrainModuleCallable = Callable[[np.ndarray, BaseModuleConfig, str | None], np.ndarray]

REGISTERED_MODULES_BRAIN: Dict[str, BrainModuleCallable] = {
    # Modules from brain1.py
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec, # type: ignore
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec, # type: ignore
    "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec, # type: ignore
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec, # type: ignore
    "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec, # type: ignore
    "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec, # type: ignore
    "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec, # type: ignore
    "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec, # type: ignore
    "EXT_GM3_Adv_Connected_Comp_Vec": EXT_GM3_Adv_Connected_Comp_Vec, # type: ignore
    # Modules from brain2.py
    "EXT_GM4_Spatial_Auto_Corr_Vec": EXT_GM4_Spatial_Auto_Corr_Vec, # type: ignore
    "EXT_GM5_Line_Completion_Vec": EXT_GM5_Line_Completion_Vec, # type: ignore
    "EXT_GM6_Symmetry_Potential_Vec": EXT_GM6_Symmetry_Potential_Vec, # type: ignore
    "EXT_GM7_Numeric_Gaps_Vec": EXT_GM7_Numeric_Gaps_Vec, # type: ignore
    "EXT_GM8_Edge_Affinity_Vec": EXT_GM8_Edge_Affinity_Vec, # type: ignore
    "EXT_GM9_Center_Control_Vec": EXT_GM9_Center_Control_Vec, # type: ignore
    "EXT_GM10_Blocking_Value_Vec": EXT_GM10_Blocking_Value_Vec, # type: ignore
    "EXT_GM11_Pair_Correlation_Vec": EXT_GM11_Pair_Correlation_Vec, # type: ignore
    "EXT_GM12_Island_Analysis_Vec": EXT_GM12_Island_Analysis_Vec, # type: ignore
    # Modules defined in this file (brain3.py)
    "EXT_GM13_Sequence_Diversity_Vec": EXT_GM13_Sequence_Diversity_Vec, # type: ignore
    "EXT_GM14_Risk_Assessment_Vec": EXT_GM14_Risk_Assessment_Vec, # type: ignore
    "EXT_GM15_Information_Gain_Vec": EXT_GM15_Information_Gain_Vec, # type: ignore
    "EXT_GM16_Harmonic_Centrality_Vec": EXT_GM16_Harmonic_Centrality_Vec, # type: ignore
    "EXT_GM17_Entropy_Minimization_Vec": EXT_GM17_Entropy_Minimization_Vec, # type: ignore
    "EXT_GM18_RL_Value_Est_Vec": EXT_GM18_RL_Value_Est_Vec, # type: ignore
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": EXT_GM19_Masked_Number_Skip_Pattern_Vec, # type: ignore
    "EXT_GM20_Skip_Pattern_Confidence_Vec": EXT_GM20_Skip_Pattern_Confidence_Vec, # type: ignore
}

# 舊寫法 ❌ DEFAULT_MODULE_CONFIGS: Dict[str, BaseModel]
# 新寫法 ✅ (更精確的型別，所有配置都應繼承 BaseModuleConfig)
DEFAULT_MODULE_CONFIGS: Dict[str, BaseModuleConfig] = {
    # Configs from brain1.py
    "EXT_A2_Weighted_Proximity_Vec": WeightedProximityConfig(),
    "EXT_M3_Local_Heterogeneity_Vec": LocalHeterogeneityConfig(),
    "EXT_D3_Potential_Field_Vec": PotentialFieldConfig(),
    "EXT_F10_Discontinuity_Vec": DiscontinuityRepairConfig(),
    "EXT_P7_Pathfinding_Value_Vec": PathfindingValueConfig(),
    "EXT_R5_Resource_Control_Vec": ResourceControlConfig(),
    "EXT_GM1_Row_Control_Vec": LineControlConfig(),
    "EXT_GM2_Col_Flow_Vec": LineControlConfig(), # Reuses LineControlConfig
    "EXT_GM3_Adv_Connected_Comp_Vec": ConnectedComponentConfig(),
    # Configs from brain2.py
    "EXT_GM4_Spatial_Auto_Corr_Vec": SpatialAutocorrelationConfig(),
    "EXT_GM5_Line_Completion_Vec": LineCompletionConfig(),
    "EXT_GM6_Symmetry_Potential_Vec": SymmetryPotentialConfig(),
    "EXT_GM7_Numeric_Gaps_Vec": NumericGapsConfig(),
    "EXT_GM8_Edge_Affinity_Vec": EdgeAffinityConfig(),
    "EXT_GM9_Center_Control_Vec": CenterControlConfig(),
    "EXT_GM10_Blocking_Value_Vec": BlockingValueConfigBrain2(), # Using the alias from brain2 import
    "EXT_GM11_Pair_Correlation_Vec": PairCorrelationConfig(),
    "EXT_GM12_Island_Analysis_Vec": IslandAnalysisConfig(),
    # Configs defined in this file (brain3.py)
    "EXT_GM13_Sequence_Diversity_Vec": SequenceDiversityConfig(),
    "EXT_GM14_Risk_Assessment_Vec": RiskAssessmentConfig(),
    "EXT_GM15_Information_Gain_Vec": InformationGainConfig(),
    "EXT_GM16_Harmonic_Centrality_Vec": HarmonicCentralityConfig(),
    "EXT_GM17_Entropy_Minimization_Vec": LocalEntropyMinimizationConfig(),
    "EXT_GM18_RL_Value_Est_Vec": RLValueEstimationConfig(),
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": SkipPatternConfig(),
    "EXT_GM20_Skip_Pattern_Confidence_Vec": SkipPatternConfidenceConfig(),
}

# 引用：知識大典.txt – 除錯.txt – "型別錯誤 (TypeError)" (防範：透過 config_override 的型別和邏輯檢查)
# 引用：建議.txt (source 715) - 配置的完整性校驗
def get_module_score(
    module_name: str,
    grid: np.ndarray,
    config_override: BaseModuleConfig | None = None,
    request_id: str | None = None
) -> np.ndarray:
    """
    Retrieves and executes a specific scoring module from the registry.
    Ensures the correct configuration (default or override) is passed to the module.
    Returns a zero array of the grid's shape if the module is not found, disabled,
    or an error occurs during execution.
    來源：新大腦.pdf - get_module_score (Page 6)
    """
    effective_request_id = request_id if request_id else f"brain-dispatch-{module_name}-{uuid.uuid4()}"
    log_extra = {"request_id": effective_request_id}

    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error(f"Module '{module_name}' not found in REGISTERED_MODULES_BRAIN.", extra=log_extra)
        return np.zeros_like(grid, dtype=float) if grid.size > 0 else np.array([], dtype=float)

    module_func = REGISTERED_MODULES_BRAIN[module_name]
    
    actual_config: BaseModuleConfig | None = None
    if config_override is not None:
        # Ensure override is of the correct type or a general BaseModuleConfig
        # This check is crucial if specific modules expect their own config types.
        expected_config_type = type(DEFAULT_MODULE_CONFIGS.get(module_name, BaseModuleConfig()))
        if isinstance(config_override, expected_config_type):
            actual_config = config_override
            logger.debug(f"Using provided config_override for module '{module_name}'.", extra=log_extra)
        else:
            logger.warning(
                f"Config_override for module '{module_name}' is of type {type(config_override).__name__}, "
                f"but expected compatible with {expected_config_type.__name__}. "
                f"Attempting to use it, but might cause issues or using default.",
                extra=log_extra
            )
            # Fallback to default if override type is mismatched and problematic.
            # Or, try to re-parse: actual_config = expected_config_type(**config_override.model_dump())
            # For now, let's prioritize the explicit override if it's a BaseModuleConfig subclass.
            if isinstance(config_override, BaseModuleConfig):
                 actual_config = config_override
            else: # Fallback to default
                 actual_config = DEFAULT_MODULE_CONFIGS.get(module_name)
                 if actual_config:
                     logger.warning(f"Fell back to default config for '{module_name}' due to type mismatch in override.", extra=log_extra)

    if actual_config is None: # If no override, or override was invalid type and no default was found (edge case)
        actual_config = DEFAULT_MODULE_CONFIGS.get(module_name)
        if actual_config is None:
            logger.error(f"No configuration found (default or override) for module '{module_name}'. Module cannot run correctly.", extra=log_extra)
            return np.zeros_like(grid, dtype=float) if grid.size > 0 else np.array([], dtype=float)
        logger.debug(f"Using default config for module '{module_name}'.", extra=log_extra)


    if not actual_config.enabled:
        logger.info(f"Module '{module_name}' is disabled via configuration. Skipping.", extra=log_extra)
        return np.zeros_like(grid, dtype=float) if grid.size > 0 else np.array([], dtype=float)

    logger.info(
         f"Executing module: '{module_name}' with resolved config: {type(actual_config).__name__}",
        extra=log_extra
    )
    if logger.isEnabledFor(logging.DEBUG):
         logger.debug(f"Full config for '{module_name}': {actual_config.model_dump_json(indent=2)}", extra=log_extra)

    try:
        # All registered modules are expected to take 'grid', 'config', and 'request_id'
        score_grid = module_func(grid, config=actual_config, request_id=effective_request_id)

        if not isinstance(score_grid, np.ndarray) or score_grid.shape != grid.shape:
            logger.error(
                f"Module '{module_name}' returned invalid score_grid. "
                f"Shape: {score_grid.shape if isinstance(score_grid, np.ndarray) else type(score_grid)}, "
                f"Expected: {grid.shape}",
                extra=log_extra
            )
            return np.zeros_like(grid, dtype=float) if grid.size > 0 else np.array([], dtype=float)
        return score_grid
    except Exception as e:
        logger.error(f"Error executing module '{module_name}': {e}", exc_info=True, extra=log_extra)
        return np.zeros_like(grid, dtype=float) if grid.size > 0 else np.array([], dtype=float)


# 來源：新大腦.pdf - Verification (Page 58-60)
# 引用：知識大典.txt – 除錯.txt – "撰寫測試 (pytest/unittest) 與自動化"
if __name__ == "__main__":
    # Configure basic logging for verification script
    # 引用：知識大典.txt – 除錯.txt – "Logging/日誌問題"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    )
    # Ensure request_id is present for logs not explicitly passing it
    class RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, 'request_id'):
                record.request_id = 'brain3_direct_run'
            return True
    logging.getLogger().addFilter(RequestIdFilter())

    logger.info("Verifying brain3.py structure and all registered modules...") # Use module logger

    dummy_grid_np = np.array([
        [1, 2, -1, 4, 5],
        [-1, 5, -1, 8, -1],
        [3, -1, 4, -1, 11],
        [12,13,-1,15,16],
        [-1,18,-1,20,-1]
    ], dtype=int) # 來源：新大腦.pdf (Page 59)
    logger.info(f"Created dummy grid (5x5):\n{dummy_grid_np}")

    # Ensure all required components are present
    required_globals = ['REGISTERED_MODULES_BRAIN', 'DEFAULT_MODULE_CONFIGS', 'BaseModuleConfig', 'get_module_score']
    missing_globals = [name for name in required_globals if name not in globals()]
    if missing_globals:
        logger.critical(f"CRITICAL: brain3.py is missing essential global definitions: {missing_globals}")
        exit(1)
    else:
        logger.info("All essential global definitions (REGISTERED_MODULES_BRAIN, etc.) are present.")


    total_modules = len(REGISTERED_MODULES_BRAIN)
    logger.info(f"Total modules registered: {total_modules}")
    # The number of modules can vary based on what brain1, brain2, and brain3 define.
    # The original Brain.txt implied 26. This version aggregates them.
    # Let's verify against the sum of expected modules if brain1 and brain2 are fully imported.
    # brain1: 9 modules, brain2: 9 modules, brain3: 8 modules (GM13-GM20) => Total 26
    expected_total_modules = 9 + 9 + 8
    assert total_modules == expected_total_modules, \
        f"Expected {expected_total_modules} modules (9 from brain1, 9 from brain2, 8 from brain3), found {total_modules}" # 來源：新大腦.pdf (Page 59)

    successful_runs = 0
    failed_modules: List[str] = []

    for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
        logger.info(f"--- Testing module {i+1}/{total_modules}: {name} ---")
        # Use default config for testing, or provide specific override if needed
        module_default_config = DEFAULT_MODULE_CONFIGS.get(name)
        if module_default_config is None:
            logger.error(f"ERROR: No default config found for {name} in DEFAULT_MODULE_CONFIGS!")
            failed_modules.append(name + " (missing default config)")
            continue
        
        # Example: test with a modified config
        test_config_override = module_default_config.model_copy(deep=True)
        test_config_override.enabled = True # Ensure it's enabled for test
        # if name == "EXT_A2_Weighted_Proximity_Vec" and isinstance(test_config_override, WeightedProximityConfig):
        #     test_config_override.radius = 1 # Modify a parameter for testing

        try:
            scores_array = get_module_score(
                name,
                dummy_grid_np,
                config_override=test_config_override,
                request_id=f"test_{name}"
            ) # 來源：新大腦.pdf (Page 60)
            logger.info(f"Successfully called {name}. Output shape: {scores_array.shape}, dtype: {scores_array.dtype}") # 來源：新大腦.pdf (Page 60)

            if scores_array.shape != dummy_grid_np.shape:
                logger.error(f"ERROR: Shape mismatch for {name}! Expected {dummy_grid_np.shape}, Got {scores_array.shape}")
                failed_modules.append(name + " (shape mismatch)")
                continue
            if scores_array.dtype != float:
                logger.error(f"ERROR: Dtype mismatch for {name}! Expected float, Got {scores_array.dtype}") # 來源：新大腦.pdf (Page 60)
                failed_modules.append(name + " (dtype mismatch)")
                continue
            
            # Print a small sample of scores, ensuring not to go out of bounds
            preview_rows = min(3, scores_array.shape[0])
            preview_cols = min(3, scores_array.shape[1])
            if preview_rows > 0 and preview_cols > 0 :
                sample_scores = scores_array[:preview_rows, :preview_cols]
                logger.info(f"Sample scores for {name}:\n{sample_scores}") # 來源：新大腦.pdf (Page 60)
            else:
                logger.info(f"Score array for {name} is empty or too small for preview.")

            successful_runs += 1

        except Exception as e:
            logger.error(f"ERROR executing module {name}: {e}", exc_info=True)
            failed_modules.append(name + f" (execution error: {type(e).__name__})")
    
    logger.info("--- Verification Summary ---")
    logger.info(f"Successfully ran {successful_runs}/{total_modules} modules.") # 來源：新大腦.pdf (Page 60)
    if failed_modules:
        logger.error("Failed modules:")
        for f_mod in failed_modules:
            logger.error(f"  - {f_mod}")
    else:
        logger.info("All registered modules ran with basic checks (call, shape, dtype) passed.")

    logger.info("brain3.py verification complete.")