# analyzer.py
# 本文件自動生成，依據新大腦.pdf、給你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 負責模組調度、分數合併與最佳建議選擇。

import numpy as np
from typing import List, Dict, Tuple, Any, Callable, Set
import logging
from pydantic import BaseModel, Field

# 來源：brain.py (本项目)
import brain 

# 來源：main.py (用户需求) - 全局统一配置日志 (Point 4.c)
logger = logging.getLogger(__name__)

# 來源：analyzer.py (用户需求 Point 3.e) - 参数定义在文件顶端
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 1 & 3
# 來源：给你2025资料在深度建议一次.pdf - 統一的配置管理 (Page 9)

class AnalyzerConfig(BaseModel):
    top_n_suggestions: int = Field(default=5, ge=1, description="返回的最佳建議數量")
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 3
    module_weights: Dict[str, float] = Field(default_factory=lambda: {
        name: 1.0 for name in brain.REGISTERED_MODULES_BRAIN.keys()
    }, description="各模組的權重")
    
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 4 & 手機優化策略 3
    enable_two_stage_filtering: bool = Field(default=True, description="是否啟用兩階段過濾")
    first_stage_candidate_count: int = Field(default=10, ge=1, description="第一階段保留的候選格數量")
    first_stage_module_names: List[str] = Field(
        default_factory=lambda: [
            "EXT_GM8_Edge_Affinity_Vec", # 假設為輕量模組
            "EXT_GM9_Center_Control_Vec", # 假設為輕量模組
        ], 
        description="第一階段使用的輕量模組名稱"
    )
    # Module-specific Pydantic configs (as designed in brain.py examples)
    # These would be loaded from a central config or set here
    weighted_proximity_config: brain.WeightedProximityConfig = Field(default_factory=brain.WeightedProximityConfig)
    discontinuity_repair_config: brain.DiscontinuityRepairConfig = Field(default_factory=brain.DiscontinuityRepairConfig)
    local_heterogeneity_config: brain.LocalHeterogeneityConfig = Field(default_factory=brain.LocalHeterogeneityConfig)
    # Add other module configs here...


DEFAULT_ANALYZER_CONFIG = AnalyzerConfig()

# 來源：analyzer.py (用户需求 Point 3.a)
# Modules are loaded from brain.py's registration
AVAILABLE_MODULES = list(brain.REGISTERED_MODULES_BRAIN.keys())

def initialize_analyzer(config_path: str | None = None):
    """
    Initializes the analyzer, potentially loading configurations or models.
    (Called by main.py on startup as per user requirement 4.b)
    """
    logger.info("Analyzer initializing...")
    if config_path:
        # Here you could load AnalyzerConfig from a YAML/JSON file if needed
        logger.info(f"Loading analyzer configuration from {config_path} (not implemented yet).")
    logger.info(f"Using default/current analyzer config: {DEFAULT_ANALYZER_CONFIG.model_dump_json(indent=2)}")
    logger.info(f"Available brain modules: {AVAILABLE_MODULES}")
    logger.info("Analyzer initialized.")


def get_module_specific_config(module_name: str, current_analyzer_config: AnalyzerConfig) -> Any | None:
    """
    Helper to retrieve the Pydantic config object for a specific module
    from the main AnalyzerConfig.
    """
    # This mapping needs to be maintained if more modules get specific Pydantic configs
    # 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 通用強化思路 1 (參數動態化)
    # 來源：给你2025资料在深度建议一次.pdf - 統一的配置管理 (Page 9)
    if module_name == "EXT_A2_Weighted_Proximity_Vec":
        return current_analyzer_config.weighted_proximity_config
    if module_name == "EXT_F10_Discontinuity_Vec":
        return current_analyzer_config.discontinuity_repair_config
    if module_name == "EXT_M3_Local_Heterogeneity_Vec":
        return current_analyzer_config.local_heterogeneity_config
    # Add other modules with specific configs here
    return None


def analyze_grid(
    grid: np.ndarray,
    request_id: str | None = None,
    analyzer_config: AnalyzerConfig = DEFAULT_ANALYZER_CONFIG,
) -> List[Dict[str, Any]]:
    """
    Analyzes the grid by dispatching to brain modules and merging scores.
    Returns a list of top N suggested cells.
    (User requirement 3.b, 3.c)
    """
    effective_request_id = request_id if request_id else "N/A_analyzer"
    logger.info(
        f"Starting grid analysis. Grid shape: {grid.shape}. Config: {analyzer_config.model_dump(mode='json', exclude_none=True)}",
        extra={"request_id": effective_request_id},
    )

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
    if analyzer_config.enable_two_stage_filtering and len(empty_cells_coords) > analyzer_config.first_stage_candidate_count:
        logger.info(f"Performing first stage filtering with modules: {analyzer_config.first_stage_module_names}", 
                    extra={"request_id": effective_request_id})
        first_stage_scores: Dict[Tuple[int, int], float] = {}
        
        for r_empty, c_empty in empty_cells_coords:
            temp_score_sum = 0.0
            num_first_stage_modules_used = 0
            for module_name in analyzer_config.first_stage_module_names:
                if module_name in brain.REGISTERED_MODULES_BRAIN:
                    module_specific_config = get_module_specific_config(module_name, analyzer_config)
                    # Ensure lightweight modules in first_stage_module_names don't require complex configs or handle None
                    score_grid = brain.get_module_score(
                        module_name, grid, config=module_specific_config, request_id=effective_request_id
                    )
                    temp_score_sum += score_grid[r_empty, c_empty] * analyzer_config.module_weights.get(module_name, 1.0)
                    num_first_stage_modules_used +=1
            
            if num_first_stage_modules_used > 0:
                 first_stage_scores[(r_empty, c_empty)] = temp_score_sum / num_first_stage_modules_used # Average score
            else:
                 first_stage_scores[(r_empty, c_empty)] = 0.0


        sorted_first_stage_cells = sorted(
            first_stage_scores.items(), key=lambda item: item[1], reverse=True
        )
        candidate_cells_coords = [
            coords for coords, score in sorted_first_stage_cells[:analyzer_config.first_stage_candidate_count]
        ]
        logger.info(f"First stage filtering selected {len(candidate_cells_coords)} candidates.", 
                    extra={"request_id": effective_request_id})


    final_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # 來源：analyzer.py (用户需求 Point 3.b) - 遍历所有空格，调用所有26个模块
    for r_empty, c_empty in candidate_cells_coords:
        aggregated_score = 0.0
        contributing_module_details: Dict[str, float] = {}
        
        # For modules not in first_stage_module_names if two-stage is enabled,
        # or all modules if not.
        modules_to_run_on_candidate = AVAILABLE_MODULES
        if analyzer_config.enable_two_stage_filtering:
            modules_to_run_on_candidate = [
                m for m in AVAILABLE_MODULES if m not in analyzer_config.first_stage_module_names
            ]
            # Add back the scores from first stage modules for these candidates
            # This logic assumes first_stage_scores holds the weighted sum for those modules.
            # A more precise aggregation would re-calculate or store individual first-stage module scores.
            # For simplicity now, we'll just sum the new module scores. A better approach would be a full re-aggregation.
            # Let's re-evaluate all modules for the selected candidates for consistency in weighting.
            modules_to_run_on_candidate = AVAILABLE_MODULES


        for module_name in modules_to_run_on_candidate:
            module_weight = analyzer_config.module_weights.get(module_name, 1.0)
            if module_weight == 0: # Skip modules with zero weight
                continue

            module_specific_config = get_module_specific_config(module_name, analyzer_config)
            
            logger.debug(f"Running module {module_name} for cell ({r_empty}, {c_empty})", 
                         extra={"request_id": effective_request_id})
            
            score_grid_for_module = brain.get_module_score(
                module_name, grid, config=module_specific_config, request_id=effective_request_id
            )
            cell_score_from_module = score_grid_for_module[r_empty, c_empty]
            
            aggregated_score += cell_score_from_module * module_weight
            contributing_module_details[module_name] = round(cell_score_from_module, 4) # Store individual score

        final_scores[(r_empty, c_empty)] = {
            "score": aggregated_score / sum(analyzer_config.module_weights.get(m, 1.0) for m in modules_to_run_on_candidate if analyzer_config.module_weights.get(m, 1.0) > 0) if sum(analyzer_config.module_weights.get(m, 1.0) for m in modules_to_run_on_candidate if analyzer_config.module_weights.get(m, 1.0) > 0) > 0 else 0, # Weighted average
            "details": contributing_module_details
        }
    
    # 來源：analyzer.py (用户需求 Point 3.c) - 选出 Top N
    sorted_suggestions = sorted(
        final_scores.items(), key=lambda item: item[1]["score"], reverse=True
    )

    top_n_results: List[Dict[str, Any]] = []
    for i in range(min(analyzer_config.top_n_suggestions, len(sorted_suggestions))):
        coords, score_info = sorted_suggestions[i]
        top_n_results.append(
            {
                "coords": coords,
                "confidence_score": round(score_info["score"], 4),
                "contributing_modules": score_info["details"] 
            }
        )
    
    logger.info(f"Analysis complete. Top {len(top_n_results)} suggestions: {top_n_results}", 
                extra={"request_id": effective_request_id})
    return top_n_results

# Example of a more complex merge strategy (conceptual)
# def adaptive_memory_merge_scores(...) -> float:
#    # 來源：analyzer.py (用户需求 Point 3.b) - 自适应记忆增强
#    # This would involve loading past game states/outcomes, feature vectors of grid, etc.
#    # For now, a weighted average is implemented.
#    pass