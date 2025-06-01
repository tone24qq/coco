# analyzer.py
# 負責模組調度、分數合併與最佳建議選擇。

import numpy as np
from typing import List, Dict, Tuple, Any, Callable, Set, Literal # Literal 新增
import logging
from pydantic import BaseModel, Field, model_validator # model_validator 新增

# 來源：brain.py (本项目)
import brain1
import brain2
import brain3
BRAIN_VERSION = os.getenv("BRAIN_VERSION", "brain2")  # 可是 brain1, brain2, brain3
brain = importlib.import_module(BRAIN_VERSION)
# 來源：main.py (用户需求 Point 4.c)
logger = logging.getLogger(__name__) # 建議使用 __name__ 以便日誌追蹤來源模組

# 強化：AnalyzerConfig 重構
class AnalyzerConfig(BaseModel):
    top_n_suggestions: int = Field(default=3, ge=1, description="返回的最佳建議數量")
    module_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            name: module_config.weight
            for name, module_config in brain.DEFAULT_MODULE_CONFIGS.items()
        },
        description="各模組的權重"
    )
    enable_two_stage_filtering: bool = Field(default=True, description="是否啟用兩階段過濾")
    first_stage_candidate_count: int = Field(default=10, ge=1, description="第一階段保留的候選格數量")
    first_stage_module_names: List[str] = Field(
        default_factory=lambda: [
            "EXT_GM8_Edge_Affinity_Vec",    # 邊緣親和度
            "EXT_GM9_Center_Control_Vec", # 中心控制偏好
            # 註：可擴展此列表，或基於 brain 模組配置中的 'cost'/'stage_preference' 元數據動態選擇。
        ],
        description="第一階段使用的輕量模組名稱"
    )
    # 新增：第一階段分數聚合策略，參考《建議.txt》
    first_stage_aggregation_strategy: Literal["average", "sum"] = Field(
        default="average",
        description="第一階段分數聚合策略 ('average' 或 'sum')。'max' 策略需修改分數累積方式。"
    )
    # 新增：最終分數合併策略 (目前僅有加權平均，為未來擴展預留)
    final_score_combination_strategy: Literal["weighted_average"] = Field(
        default="weighted_average",
        description="最終分數合併策略 (目前僅支援加權平均)"
    )

    # 強化：集中管理所有 brain 模組的 Pydantic 設定對象，參考《建議.txt》
    module_specific_configs: Dict[str, brain.BaseModuleConfig] = Field(
        default_factory=lambda: {
            name: config_instance.model_copy(deep=True) # 儲存配置實例的深拷貝以避免共享狀態
            for name, config_instance in brain.DEFAULT_MODULE_CONFIGS.items()
        },
        description="所有brain模組的具體Pydantic設定對象"
    )

    @model_validator(mode='after')
    def check_module_configs_integrity(cls, data: Any) -> Any:
        # Pydantic v2中，model_validator的第一个参数是模型实例 (self-like)
        # 或在 mode='before' 时是 dict/kwargs。这里 mode='after'，data 就是 AnalyzerConfig 实例。
        if not isinstance(data, AnalyzerConfig): # Should already be an instance
            return data

        # 確保 module_weights 和 first_stage_module_names 中的模組在 module_specific_configs 中有定義
        # 參考《建議.txt》對 AnalyzerConfig 的健壯性建議
        # Note: default_factory for module_specific_configs should ideally cover all modules
        # from brain.DEFAULT_MODULE_CONFIGS. This validator acts as a safeguard.
        for name in data.module_weights.keys():
            if name not in data.module_specific_configs:
                logger.warning(
                    f"AnalyzerConfig Integrity: Module '{name}' present in 'module_weights' but missing from "
                    f"'module_specific_configs'. It may not be configurable as expected or use a basic fallback."
                )
        
        for name in data.first_stage_module_names:
            if name not in data.module_specific_configs:
                logger.error(
                    f"AnalyzerConfig Integrity CRITICAL: Module '{name}' in 'first_stage_module_names' is missing "
                    f"from 'module_specific_configs'. This module cannot be used reliably in the first stage."
                )
            elif not data.module_specific_configs[name].enabled or data.module_weights.get(name, 0.0) == 0:
                 logger.warning(
                     f"AnalyzerConfig Integrity: First stage module '{name}' is listed but is currently disabled "
                     f"or has zero weight. It will be skipped in the first stage."
                 )
        return data

DEFAULT_ANALYZER_CONFIG = AnalyzerConfig()
ALL_AVAILABLE_MODULE_NAMES = list(brain.REGISTERED_MODULES_BRAIN.keys())

def initialize_analyzer(config_override: AnalyzerConfig | None = None) -> None:
    global DEFAULT_ANALYZER_CONFIG
    if config_override:
        DEFAULT_ANALYZER_CONFIG = config_override
        logger.info("Analyzer initialized with overridden configuration.")
    else:
        logger.info("Analyzer initialized with default configuration.")
    
    # 排除可能非常長的字典，使日誌更簡潔
    config_dump_for_log = DEFAULT_ANALYZER_CONFIG.model_dump(
        mode='json', 
        exclude={'module_weights', 'module_specific_configs'}, # 強化：排除詳細配置
        indent=2
    )
    logger.info(f"Current Analyzer Config (summary): {config_dump_for_log}")
    if logger.isEnabledFor(logging.DEBUG): # 僅在 DEBUG 級別打印完整配置
        logger.debug(f"Full Analyzer module_weights: {DEFAULT_ANALYZER_CONFIG.module_weights}")
        logger.debug(f"Full Analyzer module_specific_configs: { {k: v.model_dump_json() for k,v in DEFAULT_ANALYZER_CONFIG.module_specific_configs.items()} }")
    logger.info(f"Available brain modules: {ALL_AVAILABLE_MODULE_NAMES}")

# 強化：_get_module_specific_config_from_analyzer_config 適應新的 AnalyzerConfig 結構
def _get_module_specific_config_from_analyzer_config(
    module_name: str, analyzer_cfg: AnalyzerConfig
) -> brain.BaseModuleConfig | None:
    """
    從 AnalyzerConfig.module_specific_configs 字典中獲取特定模組的Pydantic配置對象。
    參考《建議.txt》對配置管理的建議。
    """
    module_cfg = analyzer_cfg.module_specific_configs.get(module_name)
    
    if module_cfg:
        return module_cfg
    
    # 後備邏輯：如果 module_specific_configs 中沒有（可能因為配置錯誤或 AnalyzerConfig 被手動修改），
    # 但 brain.DEFAULT_MODULE_CONFIGS 中有，則嘗試使用基本的 enabled/weight。
    if module_name in brain.DEFAULT_MODULE_CONFIGS:
        base_brain_cfg = brain.DEFAULT_MODULE_CONFIGS[module_name]
        logger.warning(
            f"Module '{module_name}' config not found in AnalyzerConfig.module_specific_configs. "
            f"Falling back to basic enabled/weight from brain.DEFAULT_MODULE_CONFIGS. "
            f"This might indicate an incomplete AnalyzerConfig setup or override."
        )
        return brain.BaseModuleConfig( # 返回一個新的基礎配置實例
            enabled=base_brain_cfg.enabled,
            weight=base_brain_cfg.weight
            # 注意：這裡只包含了基礎配置，模組特定的參數將丟失
        )
        
    logger.error(f"Configuration for module '{module_name}' not found anywhere (AnalyzerConfig or brain.DEFAULT_MODULE_CONFIGS). Cannot proceed with this module.")
    return None


def analyze_grid(
    grid: np.ndarray,
    request_id: str | None = None,
    analyzer_config_override: AnalyzerConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    透過調度 brain 模組並合併分數來分析盤面。返回 Top N 建議。
    """
    current_analyzer_config = analyzer_config_override if analyzer_config_override else DEFAULT_ANALYZER_CONFIG
    effective_request_id = request_id if request_id else "N/A_analyzer_grid"
    
    # 強化：日誌中排除 module_weights 和 module_specific_configs 以保持簡潔
    logger.info(
        f"Starting grid analysis. Grid shape: {grid.shape}. Config summary: {current_analyzer_config.model_dump(mode='json', exclude={'module_weights', 'module_specific_configs'})}",
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
    
    if current_analyzer_config.enable_two_stage_filtering and \
       len(empty_cells_coords) > current_analyzer_config.first_stage_candidate_count:
        logger.info(f"Performing first stage filtering with modules: {current_analyzer_config.first_stage_module_names} "
                    f"and strategy: {current_analyzer_config.first_stage_aggregation_strategy}",
                    extra={"request_id": effective_request_id})
        
        first_stage_cell_scores: Dict[Tuple[int, int], float] = {cell: 0.0 for cell in empty_cells_coords}
        
        active_first_stage_modules_count = 0
        for module_name in current_analyzer_config.first_stage_module_names:
            if module_name not in brain.REGISTERED_MODULES_BRAIN:
                logger.warning(f"First stage module '{module_name}' not registered in brain. Skipping.",
                               extra={"request_id": effective_request_id})
                continue

            module_specific_pydantic_config = _get_module_specific_config_from_analyzer_config(
                module_name, current_analyzer_config
            )
            
            if not module_specific_pydantic_config or not module_specific_pydantic_config.enabled:
                logger.debug(f"First stage module '{module_name}' is disabled or has no config. Skipping.",
                               extra={"request_id": effective_request_id})
                continue
            
            module_weight = current_analyzer_config.module_weights.get(module_name, 0.0) # Default to 0 if not in weights
            if module_weight == 0:
                logger.debug(f"First stage module '{module_name}' has zero weight. Skipping.",
                               extra={"request_id": effective_request_id})
                continue
            
            active_first_stage_modules_count += 1
            score_grid_for_module = brain.get_module_score(
                module_name, grid,
                config_override=module_specific_pydantic_config,
                request_id=effective_request_id
            )
            for r_empty, c_empty in empty_cells_coords:
                first_stage_cell_scores[(r_empty, c_empty)] += score_grid_for_module[r_empty, c_empty] * module_weight
        
        # 強化：應用可配置的第一階段聚合策略，參考《建議.txt》
        if active_first_stage_modules_count > 0:
            if current_analyzer_config.first_stage_aggregation_strategy == "average":
                for cell_coord_fs in first_stage_cell_scores:
                    first_stage_cell_scores[cell_coord_fs] /= active_first_stage_modules_count
            elif current_analyzer_config.first_stage_aggregation_strategy == "sum":
                pass # Scores are already summed weighted values
            # 'max' strategy would require storing individual module scores per cell for 1st stage
        
        sorted_first_stage_cells = sorted(
            first_stage_cell_scores.items(), key=lambda item: item[1], reverse=True
        )
        candidate_cells_coords = [
            coords for coords, score in sorted_first_stage_cells[:current_analyzer_config.first_stage_candidate_count]
        ]
        logger.info(f"First stage filtering selected {len(candidate_cells_coords)} candidates from {len(empty_cells_coords)} initial empty cells.",
                    extra={"request_id": effective_request_id})
    else:
        logger.info(f"Skipping two-stage filtering. Analyzing all {len(empty_cells_coords)} empty cells with all enabled modules.",
                     extra={"request_id": effective_request_id})

    final_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}
    modules_to_run_in_second_stage = [
        m_name for m_name in ALL_AVAILABLE_MODULE_NAMES
        if (m_config := _get_module_specific_config_from_analyzer_config(m_name, current_analyzer_config)) and \
           m_config.enabled and \
           current_analyzer_config.module_weights.get(m_name, 0.0) > 0
    ]
    if not modules_to_run_in_second_stage:
        logger.warning("No modules enabled or weighted for the second stage analysis. Returning empty suggestions.",
                       extra={"request_id": effective_request_id})
        return []

    logger.info(f"Second stage analysis using modules: {modules_to_run_in_second_stage} on {len(candidate_cells_coords)} candidate cells.",
                extra={"request_id": effective_request_id})

    # Pre-calculate all module scores for the grid once if many candidates, to avoid redundant calls
    # This is a significant performance optimization if candidate_cells_coords is large
    # and brain.get_module_score is expensive.
    # 參考《建議.txt》 - 中間結果的快取 (Caching) - 避免重複計算
    
    # Cache for full score grids from brain modules for the second stage
    # This avoids recomputing the same module's full grid score for each candidate cell.
    # This is a key performance enhancement for analyzer.py itself.
    module_score_cache: Dict[str, np.ndarray] = {}

    for module_name in modules_to_run_in_second_stage:
        module_specific_pydantic_config = _get_module_specific_config_from_analyzer_config(
            module_name, current_analyzer_config
        )
        # Config & weight checked when building modules_to_run_in_second_stage
        # So module_specific_pydantic_config should exist and be enabled.
        if not module_specific_pydantic_config: # Should not happen due to pre-filtering
            logger.error(f"Unexpected: Module {module_name} config missing in second stage. Skipping.", extra={"request_id": effective_request_id})
            continue

        logger.debug(f"Pre-calculating full score grid for module {module_name} (second stage)",
                     extra={"request_id": effective_request_id})
        module_score_cache[module_name] = brain.get_module_score(
            module_name, grid,
            config_override=module_specific_pydantic_config,
            request_id=effective_request_id
        )

    for r_empty, c_empty in candidate_cells_coords:
        cell_aggregated_score: float = 0.0
        total_weight_applied: float = 0.0
        contributing_module_details: Dict[str, float] = {}
        
        for module_name in modules_to_run_in_second_stage: # Iterate pre-filtered and pre-calculated modules
            # Module config, enabled status, and weight already pre-checked
            module_weight = current_analyzer_config.module_weights.get(module_name, 1.0) # Should be > 0 from pre-filter
            
            score_grid_for_module = module_score_cache.get(module_name)
            if score_grid_for_module is None: # Should not happen if pre-calculation was successful
                logger.error(f"Score grid for module {module_name} not found in cache for cell ({r_empty},{c_empty}). Skipping this module for this cell.",
                               extra={"request_id": effective_request_id})
                continue
            
            cell_score_from_module = score_grid_for_module[r_empty, c_empty]
            
            cell_aggregated_score += cell_score_from_module * module_weight
            total_weight_applied += module_weight
            contributing_module_details[module_name] = round(cell_score_from_module, 4)

        if total_weight_applied > 1e-6: # Avoid division by zero
            # Apply final score combination strategy (currently only weighted_average)
            if current_analyzer_config.final_score_combination_strategy == "weighted_average":
                 final_cell_score = cell_aggregated_score / total_weight_applied
            else: # Fallback or if other strategies are added
                 final_cell_score = cell_aggregated_score / total_weight_applied # Default to weighted_average
        else:
            final_cell_score = 0.0 # Or handle as an error/low confidence
            
        final_scores[(r_empty, c_empty)] = {
            "score": final_cell_score,
            "details": contributing_module_details
        }
    
    sorted_suggestions = sorted(
        final_scores.items(), key=lambda item: item[1]["score"], reverse=True
    )

    top_n_results: List[Dict[str, Any]] = []
    for i in range(min(current_analyzer_config.top_n_suggestions, len(sorted_suggestions))):
        coords, score_info = sorted_suggestions[i]
        top_n_results.append(
            {
                "coords": coords,
                "confidence_score": round(score_info["score"], 4),
                "contributing_modules": score_info["details"]
            }
        )
    
    logger.info(f"Analysis complete. Top {len(top_n_results)} suggestions generated.",
                extra={"request_id": effective_request_id})
    if logger.isEnabledFor(logging.DEBUG):
         logger.debug(f"Top suggestions details: {top_n_results}", extra={"request_id": effective_request_id})
    return top_n_results