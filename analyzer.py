# analyzer.py
# 本文件自動生成，依據新大腦.pdf、给你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 負責模組調度、分數合併與最佳建議選擇。

import numpy as np
from typing import List, Dict, Tuple, Any, Callable, Set
import logging
from pydantic import BaseModel, Field

# 來源：brain.py (本项目)
import brain

# 來源：main.py (用户需求 Point 4.c)
logger = logging.getLogger(__name__)

# 來源：analyzer.py (用户需求 Point 3.e) - 参数定义在文件顶端
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 1 & 3
# 來源：给你2025资料在深度建议一次.pdf - 統一的配置管理 (Page 9)

class AnalyzerConfig(BaseModel):
    top_n_suggestions: int = Field(default=3, ge=1, description="返回的最佳建議數量")
    # 預設所有模組權重為1.0，可以在main.py中或透過設定檔修改
    module_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            name: module_config.weight 
            for name, module_config in brain.DEFAULT_MODULE_CONFIGS.items()
        },
        description="各模組的權重"
    )
    
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 4 & 手機優化策略 3
    enable_two_stage_filtering: bool = Field(default=True, description="是否啟用兩階段過濾")
    first_stage_candidate_count: int = Field(default=10, ge=1, description="第一階段保留的候選格數量")
    # 假設這些是計算開銷較小的模組
    first_stage_module_names: List[str] = Field(
        default_factory=lambda: [
            "EXT_GM8_Edge_Affinity_Vec",    # 邊緣親和度
            "EXT_GM9_Center_Control_Vec", # 中心控制偏好
            # 可以再加入其他計算快的模組，例如只看直接鄰居的簡單模組
        ], 
        description="第一階段使用的輕量模組名稱"
    )

    # 包含所有 brain 模組的 Pydantic 設定物件
    # 這樣可以集中管理，並在需要時動態調整 analyzer 傳遞給 brain 模組的設定
    # 來源：给你2025资料在深度建议一次.pdf - 統一的配置管理 (Page 9)
    # 每個模組的設定都從 brain.py 中對應的 Pydantic 模型實例化
    ext_a2_weighted_proximity_vec_config: brain.WeightedProximityConfig = Field(default_factory=brain.WeightedProximityConfig)
    ext_m3_local_heterogeneity_vec_config: brain.LocalHeterogeneityConfig = Field(default_factory=brain.LocalHeterogeneityConfig)
    ext_d3_potential_field_vec_config: brain.PotentialFieldConfig = Field(default_factory=brain.PotentialFieldConfig)
    ext_f10_discontinuity_vec_config: brain.DiscontinuityRepairConfig = Field(default_factory=brain.DiscontinuityRepairConfig)
    ext_p7_pathfinding_value_vec_config: brain.PathfindingValueConfig = Field(default_factory=brain.PathfindingValueConfig)
    ext_r5_resource_control_vec_config: brain.ResourceControlConfig = Field(default_factory=brain.ResourceControlConfig)
    ext_gm1_row_control_vec_config: brain.LineControlConfig = Field(default_factory=brain.LineControlConfig)
    ext_gm2_col_flow_vec_config: brain.LineControlConfig = Field(default_factory=brain.LineControlConfig)
    ext_gm3_adv_connected_comp_vec_config: brain.ConnectedComponentConfig = Field(default_factory=brain.ConnectedComponentConfig)
    ext_gm4_spatial_auto_corr_vec_config: brain.SpatialAutocorrelationConfig = Field(default_factory=brain.SpatialAutocorrelationConfig)
    ext_gm5_line_completion_vec_config: brain.LineCompletionConfig = Field(default_factory=brain.LineCompletionConfig)
    ext_gm6_symmetry_potential_vec_config: brain.SymmetryPotentialConfig = Field(default_factory=brain.SymmetryPotentialConfig)
    ext_gm7_numeric_gaps_vec_config: brain.NumericGapsConfig = Field(default_factory=brain.NumericGapsConfig)
    ext_gm8_edge_affinity_vec_config: brain.EdgeAffinityConfig = Field(default_factory=brain.EdgeAffinityConfig)
    ext_gm9_center_control_vec_config: brain.CenterControlConfig = Field(default_factory=brain.CenterControlConfig)
    ext_gm10_blocking_value_vec_config: brain.BlockingValueConfig = Field(default_factory=brain.BlockingValueConfig)
    ext_gm11_pair_correlation_vec_config: brain.PairCorrelationConfig = Field(default_factory=brain.PairCorrelationConfig)
    ext_gm12_island_analysis_vec_config: brain.IslandAnalysisConfig = Field(default_factory=brain.IslandAnalysisConfig)
    ext_gm13_sequence_diversity_vec_config: brain.SequenceDiversityConfig = Field(default_factory=brain.SequenceDiversityConfig)
    ext_gm14_risk_assessment_vec_config: brain.RiskAssessmentConfig = Field(default_factory=brain.RiskAssessmentConfig)
    ext_gm15_information_gain_vec_config: brain.InformationGainConfig = Field(default_factory=brain.InformationGainConfig)
    ext_gm16_harmonic_centrality_vec_config: brain.HarmonicCentralityConfig = Field(default_factory=brain.HarmonicCentralityConfig)
    ext_gm17_entropy_minimization_vec_config: brain.LocalEntropyMinimizationConfig = Field(default_factory=brain.LocalEntropyMinimizationConfig)
    ext_gm18_rl_value_est_vec_config: brain.RLValueEstimationConfig = Field(default_factory=brain.RLValueEstimationConfig)
    ext_gm19_masked_number_skip_pattern_vec_config: brain.SkipPatternConfig = Field(default_factory=brain.SkipPatternConfig)
    ext_gm20_skip_pattern_confidence_vec_config: brain.SkipPatternConfidenceConfig = Field(default_factory=brain.SkipPatternConfidenceConfig)

DEFAULT_ANALYZER_CONFIG = AnalyzerConfig()

# 來源：analyzer.py (用户需求 Point 3.a)
# Modules are loaded from brain.py's registration
ALL_AVAILABLE_MODULE_NAMES = list(brain.REGISTERED_MODULES_BRAIN.keys())

def initialize_analyzer(config_override: AnalyzerConfig | None = None) -> None:
    """
    Initializes the analyzer.
    (Called by main.py on startup as per user requirement 4.b)
    """
    global DEFAULT_ANALYZER_CONFIG
    if config_override:
        DEFAULT_ANALYZER_CONFIG = config_override
        logger.info(f"Analyzer initialized with overridden configuration.")
    else:
        logger.info(f"Analyzer initialized with default configuration.")
    
    logger.info(f"Current Analyzer Config: {DEFAULT_ANALYZER_CONFIG.model_dump_json(indent=2, exclude_none=True)}")
    logger.info(f"Available brain modules: {ALL_AVAILABLE_MODULE_NAMES}")


def _get_module_specific_config_from_analyzer_config(
    module_name: str, analyzer_cfg: AnalyzerConfig
) -> brain.BaseModuleConfig | None:
    """
    Helper to retrieve the Pydantic config object for a specific module
    from the main AnalyzerConfig.
    """
    # Maps module name to its attribute name in AnalyzerConfig
    # This mapping needs to be maintained manually if AnalyzerConfig field names change
    config_attr_name = module_name.lower() + "_config"
    if hasattr(analyzer_cfg, config_attr_name):
        return getattr(analyzer_cfg, config_attr_name)
    
    # Fallback for modules that might only have BaseModuleConfig (enabled/weight)
    # or if a specific config field was missed in AnalyzerConfig.
    # This should ideally not be hit if AnalyzerConfig is complete.
    if module_name in brain.DEFAULT_MODULE_CONFIGS:
        # Return a basic config with just enabled/weight if specific one is missing in AnalyzerConfig
        base_cfg = brain.BaseModuleConfig(
            enabled=brain.DEFAULT_MODULE_CONFIGS[module_name].enabled,
            weight=brain.DEFAULT_MODULE_CONFIGS[module_name].weight
        )
        logger.warning(f"Specific config for {module_name} not found directly in AnalyzerConfig, using its default BaseModuleConfig values for enabled/weight. AnalyzerConfig might need update.")
        return base_cfg
        
    logger.warning(f"No specific config attribute or default found for {module_name} in AnalyzerConfig or brain.DEFAULT_MODULE_CONFIGS.")
    return None


def analyze_grid(
    grid: np.ndarray,
    request_id: str | None = None,
    analyzer_config_override: AnalyzerConfig | None = None, # Allow overriding the global default
) -> List[Dict[str, Any]]:
    """
    Analyzes the grid by dispatching to brain modules and merging scores.
    Returns a list of top N suggested cells.
    (User requirement 3.b, 3.c)
    """
    current_analyzer_config = analyzer_config_override if analyzer_config_override else DEFAULT_ANALYZER_CONFIG
    effective_request_id = request_id if request_id else "N/A_analyzer"
    
    logger.info(
        f"Starting grid analysis. Grid shape: {grid.shape}. Config in use: {current_analyzer_config.model_dump(mode='json', exclude={'module_weights'})}", # Exclude potentially long weights dict
        extra={"request_id": effective_request_id},
    )
    # logger.debug(f"Full module weights: {current_analyzer_config.module_weights}", extra={"request_id": effective_request_id})


    rows, cols = grid.shape
    empty_cells_coords: List[Tuple[int, int]] = [
        (r, c) for r in range(rows) for c in range(cols) if grid[r, c] == -1
    ]

    if not empty_cells_coords:
        logger.info("No empty cells to analyze.", extra={"request_id": effective_request_id})
        return []

    candidate_cells_coords = empty_cells_coords
    
    # 來源：analyzer.py (用户需求 Point 5.c)
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 手機優化策略 3 (分階段評估)
    if current_analyzer_config.enable_two_stage_filtering and \
       len(empty_cells_coords) > current_analyzer_config.first_stage_candidate_count:
        logger.info(f"Performing first stage filtering with modules: {current_analyzer_config.first_stage_module_names}", 
                    extra={"request_id": effective_request_id})
        
        first_stage_cell_scores: Dict[Tuple[int, int], float] = {cell: 0.0 for cell in empty_cells_coords}
        
        for module_name in current_analyzer_config.first_stage_module_names:
            if module_name not in brain.REGISTERED_MODULES_BRAIN:
                logger.warning(f"First stage module {module_name} not found in brain. Skipping.",
                               extra={"request_id": effective_request_id})
                continue

            module_specific_pydantic_config = _get_module_specific_config_from_analyzer_config(
                module_name, current_analyzer_config
            )
            if not module_specific_pydantic_config or not module_specific_pydantic_config.enabled:
                logger.debug(f"First stage module {module_name} is disabled or has no config. Skipping.",
                               extra={"request_id": effective_request_id})
                continue
            
            module_weight = current_analyzer_config.module_weights.get(module_name, 1.0)
            if module_weight == 0: continue

            score_grid_for_module = brain.get_module_score(
                module_name, grid, 
                config_override=module_specific_pydantic_config, # Pass the specific config
                request_id=effective_request_id
            )
            for r_empty, c_empty in empty_cells_coords:
                first_stage_cell_scores[(r_empty, c_empty)] += score_grid_for_module[r_empty, c_empty] * module_weight
        
        # Average scores if multiple first-stage modules (or use sum, depending on strategy)
        num_first_stage_active_modules = sum(
            1 for mn in current_analyzer_config.first_stage_module_names 
            if current_analyzer_config.module_weights.get(mn, 1.0) > 0 and 
               (_get_module_specific_config_from_analyzer_config(mn, current_analyzer_config) and 
                _get_module_specific_config_from_analyzer_config(mn, current_analyzer_config).enabled) # type: ignore
        )
        
        if num_first_stage_active_modules > 0:
            for cell_coord in first_stage_cell_scores:
                first_stage_cell_scores[cell_coord] /= num_first_stage_active_modules # Average the sum

        sorted_first_stage_cells = sorted(
            first_stage_cell_scores.items(), key=lambda item: item[1], reverse=True
        )
        candidate_cells_coords = [
            coords for coords, score in sorted_first_stage_cells[:current_analyzer_config.first_stage_candidate_count]
        ]
        logger.info(f"First stage filtering selected {len(candidate_cells_coords)} candidates from {len(empty_cells_coords)}.", 
                    extra={"request_id": effective_request_id})
    else:
        logger.info(f"Skipping two-stage filtering. Analyzing all {len(empty_cells_coords)} empty cells.",
                     extra={"request_id": effective_request_id})


    final_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # 來源：analyzer.py (用户需求 Point 3.b) - 遍历所有空格，调用所有26个模块 (now candidate_cells_coords)
    for r_empty, c_empty in candidate_cells_coords:
        cell_aggregated_score: float = 0.0
        total_weight_applied: float = 0.0
        contributing_module_details: Dict[str, float] = {}
        
        for module_name in ALL_AVAILABLE_MODULE_NAMES: # Iterate all available modules for the second stage
            module_specific_pydantic_config = _get_module_specific_config_from_analyzer_config(
                module_name, current_analyzer_config
            )

            if not module_specific_pydantic_config or not module_specific_pydantic_config.enabled:
                logger.debug(f"Module {module_name} is disabled or has no config for cell ({r_empty},{c_empty}). Skipping.",
                               extra={"request_id": effective_request_id})
                continue

            module_weight = current_analyzer_config.module_weights.get(module_name, 1.0)
            if module_weight == 0: # Skip modules with zero weight
                logger.debug(f"Module {module_name} has zero weight for cell ({r_empty},{c_empty}). Skipping.",
                               extra={"request_id": effective_request_id})
                continue
            
            logger.debug(f"Running module {module_name} for cell ({r_empty}, {c_empty}) with weight {module_weight}", 
                         extra={"request_id": effective_request_id})
            
            score_grid_for_module = brain.get_module_score(
                module_name, grid, 
                config_override=module_specific_pydantic_config, 
                request_id=effective_request_id
            )
            cell_score_from_module = score_grid_for_module[r_empty, c_empty]
            
            cell_aggregated_score += cell_score_from_module * module_weight
            total_weight_applied += module_weight
            contributing_module_details[module_name] = round(cell_score_from_module, 4)

        if total_weight_applied > 1e-6 : # Avoid division by zero if all weights were zero
            final_cell_score = cell_aggregated_score / total_weight_applied # Weighted average
        else:
            final_cell_score = 0.0
            
        final_scores[(r_empty, c_empty)] = {
            "score": final_cell_score,
            "details": contributing_module_details
        }
    
    # 來源：analyzer.py (用户需求 Point 3.c) - 选出 Top N
    sorted_suggestions = sorted(
        final_scores.items(), key=lambda item: item[1]["score"], reverse=True
    )

    top_n_results: List[Dict[str, Any]] = []
    for i in range(min(current_analyzer_config.top_n_suggestions, len(sorted_suggestions))):
        coords, score_info = sorted_suggestions[i]
        top_n_results.append(
            {
                "coords": coords, # Tuple[int, int]
                "confidence_score": round(score_info["score"], 4),
                "contributing_modules": score_info["details"] 
            }
        )
    
    logger.info(f"Analysis complete. Top {len(top_n_results)} suggestions: {top_n_results}", 
                extra={"request_id": effective_request_id})
    return top_n_results