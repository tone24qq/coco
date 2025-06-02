# analyzer.py
# 負責模組調度、分數合併與最佳建議選擇。

# 來源：知識大典.txt – 防錯字典.txt – "PEP 8 代码风格指南" – "導入順序"
# 1. 標準庫導入
import importlib
import logging
import os
from typing import Any, Callable, Dict, List, Literal, Set, Tuple # Literal for Pydantic V2+
import brain1
import brain2
import brain3
# 選擇要用哪個大腦
BRAIN_VERSION = os.environ.get("BRAIN_VERSION", "brain3")  # 預設用 brain3.py
brain = importlib.import_module(BRAIN_VERSION)
# 2. 第三方庫導入
import numpy as np
from pydantic import BaseModel, Field, model_validator

# 3. 本地應用/自定义模块导入
# Dynamically import the specified brain module (brain1, brain2, or brain3)
# This allows for flexible selection of the underlying logic.
# 引用：建議.txt - "AI 學習的重點：Numba/Cython 適用場景" (間接，brain模組可能使用這些技術)
BRAIN_VERSION_DEFAULT = "brain2" # Default if environment variable is not set
BRAIN_VERSION = os.getenv("BRAIN_VERSION", BRAIN_VERSION_DEFAULT)
try:
    # 來源：知識大典.txt – 防錯字典.txt – "ImportError" (防範：嘗試導入並處理可能的錯誤)
    brain = importlib.import_module(f"brain{BRAIN_VERSION.replace('brain', '')}")
    # Ensure the imported brain module has the necessary components
    if not all(hasattr(brain, attr) for attr in ['DEFAULT_MODULE_CONFIGS', 'REGISTERED_MODULES_BRAIN', 'BaseModuleConfig', 'get_module_score']):
        raise ImportError(f"Module 'brain{BRAIN_VERSION.replace('brain', '')}' is missing required attributes.")
except ImportError as e:
    logging.error(f"Failed to import brain module version '{BRAIN_VERSION}': {e}. Falling back to '{BRAIN_VERSION_DEFAULT}'.", exc_info=True)
    try:
        brain = importlib.import_module(BRAIN_VERSION_DEFAULT) # Use f-string for consistency if needed: f"brain{BRAIN_VERSION_DEFAULT.replace('brain', '')}"
        if not all(hasattr(brain, attr) for attr in ['DEFAULT_MODULE_CONFIGS', 'REGISTERED_MODULES_BRAIN', 'BaseModuleConfig', 'get_module_score']):
            raise ImportError(f"Fallback brain module '{BRAIN_VERSION_DEFAULT}' is also missing required attributes. Analyzer cannot function.")
    except ImportError as e_fallback:
        logging.critical(f"CRITICAL: Failed to import fallback brain3 module '{BRAIN_VERSION_DEFAULT}': {e_fallback}. Analyzer will not work.", exc_info=True)
        # Depending on application requirements, might raise SystemExit here
        raise # Re-raise the critical error to prevent app from starting in a broken state

# --- Logging Setup ---
# 引用：知識大典.txt – 除錯.txt – "Logging/日誌問題" – "記錄 request_id/trace_id" (logger應配合main.py的RequestIdLoggerAdapter)
logger = logging.getLogger(__name__) # Use __name__ for module-specific logging context

# --- Analyzer Configuration Model ---
# 引用：建議.txt – "針對 analyzer.py 的深入建議與強化" – "配置管理的健壯性與動態性"
class AnalyzerConfig(BaseModel):
    """
    Configuration for the analyzer, controlling how suggestions are generated.
    Includes module weights, filtering strategies, and specific configurations for brain modules.
    """
    top_n_suggestions: int = Field(default=3, ge=1, description="返回的最佳建議數量")
    module_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            name: module_config.weight
            for name, module_config in brain.DEFAULT_MODULE_CONFIGS.items() # type: ignore[attr-defined]
        },
        description="各模組的權重"
    )
    enable_two_stage_filtering: bool = Field(default=True, description="是否啟用兩階段過濾")
    first_stage_candidate_count: int = Field(default=10, ge=1, description="第一階段保留的候選格數量")
    first_stage_module_names: List[str] = Field(
        default_factory=lambda: [
            "EXT_GM8_Edge_Affinity_Vec",
            "EXT_GM9_Center_Control_Vec",
        ],
        description="第一階段使用的輕量模組名稱。可擴展或基於模組元數據動態選擇。"
    )
    first_stage_aggregation_strategy: Literal["average", "sum"] = Field(
        default="average",
        description="第一階段分數聚合策略 ('average' 或 'sum')."
    )
    final_score_combination_strategy: Literal["weighted_average"] = Field(
        default="weighted_average",
        description="最終分數合併策略 (目前僅支援加權平均)."
    )
    module_specific_configs: Dict[str, brain.BaseModuleConfig] = Field( # type: ignore[attr-defined]
        default_factory=lambda: {
            name: config_instance.model_copy(deep=True)
            for name, config_instance in brain.DEFAULT_MODULE_CONFIGS.items() # type: ignore[attr-defined]
        },
        description="所有brain模組的具體Pydantic設定對象的深拷貝。"
    )

    # 引用：建議.txt - "針對 analyzer.py 的深入建議與強化" - "配置管理的健壯性與動態性"
    # 引用：知識大典.txt – 除錯.txt – "Logging/日誌問題" (日誌記錄配置問題)
    @model_validator(mode='after')
    def check_module_configs_integrity(self) -> 'AnalyzerConfig':
        """
        Validates the integrity of module configurations within AnalyzerConfig.
        Ensures that modules listed in weights and stages have corresponding configs.
        """
        # Pydantic V2: 'self' is the instance of AnalyzerConfig for mode='after'
        for name in self.module_weights.keys():
            if name not in self.module_specific_configs:
                logger.warning(
                    f"AnalyzerConfig Integrity: Module '{name}' in 'module_weights' missing from "
                    f"'module_specific_configs'. May use basic fallback or defaults."
                )

        for name in self.first_stage_module_names:
            if name not in self.module_specific_configs:
                logger.error(
                    f"AnalyzerConfig Integrity CRITICAL: Module '{name}' in 'first_stage_module_names' "
                    f"is missing from 'module_specific_configs'. Cannot reliably use in first stage."
                )
            else:
                module_cfg = self.module_specific_configs[name]
                # 舊寫法 ❌ (直接比較 module_cfg.enabled and self.module_weights.get(name, 0.0) == 0)
                # 新寫法 ✅ (更清晰的條件分離)
                if not module_cfg.enabled:
                    logger.warning(
                        f"AnalyzerConfig Integrity: First stage module '{name}' is listed but currently disabled."
                    )
                elif self.module_weights.get(name, 0.0) == 0.0: # Use 0.0 for float comparison
                    logger.warning(
                        f"AnalyzerConfig Integrity: First stage module '{name}' is listed but has zero weight."
                    )
        return self

# --- Global Analyzer State ---
DEFAULT_ANALYZER_CONFIG = AnalyzerConfig()
# 來源：知識大典.txt – 防錯字典.txt – "KeyError" (防範：確保鍵存在於 brain.REGISTERED_MODULES_BRAIN)
ALL_AVAILABLE_MODULE_NAMES: List[str] = list(brain.REGISTERED_MODULES_BRAIN.keys()) # type: ignore[attr-defined]

def initialize_analyzer(config_override: AnalyzerConfig | None = None) -> None:
    """
    Initializes the analyzer with default or overridden configuration.
    Logs the configuration summary.
    """
    global DEFAULT_ANALYZER_CONFIG
    if config_override:
        DEFAULT_ANALYZER_CONFIG = config_override
        logger.info("Analyzer initialized with overridden configuration.")
    else:
        logger.info("Analyzer initialized with default configuration.")

    config_dump_for_log = DEFAULT_ANALYZER_CONFIG.model_dump(
        mode='json',
        exclude={'module_weights', 'module_specific_configs'}, # Exclude verbose fields
        indent=2
    )
    logger.info(f"Current Analyzer Config (summary): {config_dump_for_log}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Full Analyzer module_weights: {DEFAULT_ANALYZER_CONFIG.module_weights}")
        # Log specific module configs carefully if they contain sensitive or very large data
        debug_module_configs = {
            k: v.model_dump(exclude_none=True) for k, v in DEFAULT_ANALYZER_CONFIG.module_specific_configs.items()
        }
        logger.debug(f"Full Analyzer module_specific_configs (non-default values): {debug_module_configs}")
    logger.info(f"Available brain modules from '{BRAIN_VERSION}': {ALL_AVAILABLE_MODULE_NAMES}")

# 引用：建議.txt - "針對 analyzer.py 的深入建議與強化" - "配置管理的健壯性與動態性"
def _get_module_specific_config_from_analyzer_config(
    module_name: str, analyzer_cfg: AnalyzerConfig
) -> brain.BaseModuleConfig | None: # type: ignore[attr-defined]
    """
    Retrieves the Pydantic configuration object for a specific module from AnalyzerConfig.
    Falls back to basic enabled/weight from brain.DEFAULT_MODULE_CONFIGS if not found in AnalyzerConfig.
    """
    module_cfg = analyzer_cfg.module_specific_configs.get(module_name)
    if module_cfg:
        return module_cfg

    # Fallback logic for robustness, though ideally AnalyzerConfig should be complete.
    # 引用：知識大典.txt – 防錯字典.txt – "KeyError" (防範：使用 .get() 或檢查鍵是否存在)
    if module_name in brain.DEFAULT_MODULE_CONFIGS: # type: ignore[attr-defined]
        base_brain_cfg = brain.DEFAULT_MODULE_CONFIGS[module_name] # type: ignore[attr-defined]
        logger.warning(
            f"Module '{module_name}' config not in AnalyzerConfig.module_specific_configs. "
            f"Falling back to basic enabled/weight from brain.DEFAULT_MODULE_CONFIGS for '{BRAIN_VERSION}'. "
            f"This may indicate incomplete AnalyzerConfig or override issues."
        )
        # Return a new instance of BaseModuleConfig with basic settings
        return brain.BaseModuleConfig(enabled=base_brain_cfg.enabled, weight=base_brain_cfg.weight) # type: ignore[attr-defined]

    logger.error(f"Configuration for module '{module_name}' not found in AnalyzerConfig or brain.DEFAULT_MODULE_CONFIGS. Module cannot be used.")
    return None


# 引用：建議.txt - "針對 analyzer.py 的深入建議與強化" - "兩階段過濾策略的增強", "分數合併與建議選擇的精緻化"
# 引用：建議.txt - "中間結果的快取 (Caching)"
def analyze_grid(
    grid: np.ndarray,
    request_id: str | None = None,
    analyzer_config_override: AnalyzerConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    Analyzes the given grid by dispatching to brain modules and combining their scores.
    Returns a list of top N suggestions with coordinates, confidence scores, and contributing modules.

    Args:
        grid: A NumPy array representing the game board.
        request_id: Optional request identifier for logging.
        analyzer_config_override: Optional AnalyzerConfig to override the default.

    Returns:
        A list of dictionaries, each representing a suggestion.
    """
    current_analyzer_config = analyzer_config_override if analyzer_config_override else DEFAULT_ANALYZER_CONFIG
    effective_request_id = request_id if request_id else f"analyzer-{uuid.uuid4()}" # Ensure a request_id

    log_extra = {"request_id": effective_request_id}
    logger.info(
        f"Starting grid analysis. Grid shape: {grid.shape}. Config summary: "
        f"{current_analyzer_config.model_dump(mode='json', exclude={'module_weights', 'module_specific_configs'})}",
        extra=log_extra,
    )

    rows, cols = grid.shape
    # 舊寫法 ❌ (List comprehension might be less readable for complex conditions or many items)
    # empty_cells_coords: List[Tuple[int, int]] = [
    #     (r, c) for r in range(rows) for c in range(cols) if grid[r, c] == -1
    # ]
    # 新寫法 ✅ (Using np.argwhere for potentially better performance and conciseness on large grids)
    # 引用：建議.txt - "NumPy 向量化 (Vectorization)" - "條件計算"
    empty_cells_indices = np.argwhere(grid == -1)
    empty_cells_coords: List[Tuple[int, int]] = [tuple(coords) for coords in empty_cells_indices.tolist()]


    if not empty_cells_coords:
        logger.info("No empty cells to analyze.", extra=log_extra)
        return []

    candidate_cells_coords = empty_cells_coords
    module_score_cache: Dict[str, np.ndarray] = {} # Cache for full score grids

    # --- First Stage Filtering (if enabled and applicable) ---
    if current_analyzer_config.enable_two_stage_filtering and \
       len(empty_cells_coords) > current_analyzer_config.first_stage_candidate_count:
        logger.info(f"Performing first stage filtering with modules: {current_analyzer_config.first_stage_module_names} "
                    f"and strategy: {current_analyzer_config.first_stage_aggregation_strategy}",
                    extra=log_extra)

        first_stage_cell_scores: Dict[Tuple[int, int], float] = {cell: 0.0 for cell in empty_cells_coords}
        active_first_stage_modules_count = 0

        for module_name in current_analyzer_config.first_stage_module_names:
            if module_name not in brain.REGISTERED_MODULES_BRAIN: # type: ignore[attr-defined]
                logger.warning(f"First stage module '{module_name}' not registered. Skipping.", extra=log_extra)
                continue

            module_specific_cfg = _get_module_specific_config_from_analyzer_config(module_name, current_analyzer_config)
            module_weight = current_analyzer_config.module_weights.get(module_name, 0.0)

            if not module_specific_cfg or not module_specific_cfg.enabled or module_weight == 0.0:
                logger.debug(f"Skipping disabled/zero-weight/unconfigured first stage module '{module_name}'.", extra=log_extra)
                continue

            active_first_stage_modules_count += 1
            # 引用：建議.txt - "中間結果的快取 (Caching)" (雖然此處是第一階段，但get_module_score本身應高效)
            score_grid = brain.get_module_score(module_name, grid, config_override=module_specific_cfg, request_id=effective_request_id) # type: ignore[attr-defined]
            for r_empty, c_empty in empty_cells_coords:
                first_stage_cell_scores[(r_empty, c_empty)] += score_grid[r_empty, c_empty] * module_weight
        
        if active_first_stage_modules_count > 0:
            if current_analyzer_config.first_stage_aggregation_strategy == "average":
                for cell_coord_fs in first_stage_cell_scores: # Iterate over keys
                    first_stage_cell_scores[cell_coord_fs] /= active_first_stage_modules_count
            # For "sum", scores are already weighted sums.
        
        sorted_first_stage_cells = sorted(first_stage_cell_scores.items(), key=lambda item: item[1], reverse=True)
        candidate_cells_coords = [
            coords for coords, score in sorted_first_stage_cells[:current_analyzer_config.first_stage_candidate_count]
        ]
        logger.info(f"First stage filtering selected {len(candidate_cells_coords)} candidates from {len(empty_cells_coords)} empty cells.",
                    extra=log_extra)
    else:
        logger.info(f"Skipping two-stage filtering. Analyzing all {len(empty_cells_coords)} empty cells.", extra=log_extra)

    # --- Second Stage Analysis (on candidate cells) ---
    final_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}
    modules_to_run_second_stage = [
        m_name for m_name in ALL_AVAILABLE_MODULE_NAMES
        if (m_config := _get_module_specific_config_from_analyzer_config(m_name, current_analyzer_config)) and \
           m_config.enabled and \
           current_analyzer_config.module_weights.get(m_name, 0.0) > 0.0
    ]

    if not modules_to_run_second_stage:
        logger.warning("No modules enabled or weighted for second stage analysis. Returning empty.", extra=log_extra)
        return []

    logger.info(f"Second stage analysis using modules: {modules_to_run_second_stage} on {len(candidate_cells_coords)} candidates.",
                extra=log_extra)

    # Pre-calculate all required module score grids for the second stage
    # 引用：建議.txt - "中間結果的快取 (Caching)" - 避免重複計算 (analyzer.py 的核心優化)
    for module_name in modules_to_run_second_stage:
        if module_name not in module_score_cache: # Only compute if not already cached (e.g. from a hypothetical shared cache)
            module_specific_cfg = _get_module_specific_config_from_analyzer_config(module_name, current_analyzer_config)
            if not module_specific_cfg: # Should have been caught by list comprehension filter
                logger.error(f"Unexpected: Config for second stage module {module_name} missing. Skipping pre-calculation.", extra=log_extra)
                continue
            logger.debug(f"Pre-calculating full score grid for second stage module: {module_name}", extra=log_extra)
            module_score_cache[module_name] = brain.get_module_score( # type: ignore[attr-defined]
                module_name, grid, config_override=module_specific_cfg, request_id=effective_request_id
            )

    for r_empty, c_empty in candidate_cells_coords:
        cell_aggregated_score: float = 0.0
        total_weight_applied: float = 0.0
        contributing_module_details: Dict[str, float] = {}

        for module_name in modules_to_run_second_stage:
            module_weight = current_analyzer_config.module_weights.get(module_name, 1.0) # Already filtered for >0 weight
            
            score_grid_for_module = module_score_cache.get(module_name)
            if score_grid_for_module is None:
                logger.error(f"Score grid for {module_name} not found in cache for cell ({r_empty},{c_empty}). Skipping.",
                               extra=log_extra)
                continue
            
            cell_score_from_module = score_grid_for_module[r_empty, c_empty]
            cell_aggregated_score += cell_score_from_module * module_weight
            total_weight_applied += module_weight
            contributing_module_details[module_name] = round(cell_score_from_module, 4)

        final_cell_score = 0.0
        # 引用：知識大典.txt – 防錯字典.txt – "ArithmeticError" (防範 ZeroDivisionError)
        if abs(total_weight_applied) > 1e-9: # Use abs for safety, though weights should be positive
            if current_analyzer_config.final_score_combination_strategy == "weighted_average":
                final_cell_score = cell_aggregated_score / total_weight_applied
            # Add other strategies here if implemented
            else: # Fallback to weighted_average
                 final_cell_score = cell_aggregated_score / total_weight_applied
        elif cell_aggregated_score != 0.0 : # Weights are zero, but score is not. Log this anomaly.
             logger.warning(f"Cell ({r_empty},{c_empty}) has non-zero aggregated score ({cell_aggregated_score}) but total_weight_applied is near zero. Setting final score to 0.", extra=log_extra)

        final_scores[(r_empty, c_empty)] = {
            "score": final_cell_score,
            "details": contributing_module_details
        }
    
    # Sort suggestions by final score
    sorted_suggestions = sorted(final_scores.items(), key=lambda item: item[1]["score"], reverse=True)

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
    
    logger.info(f"Analysis complete. Top {len(top_n_results)} suggestions generated.", extra=log_extra)
    if logger.isEnabledFor(logging.DEBUG):
         logger.debug(f"Top suggestions details: {top_n_results}", extra=log_extra)
    return top_n_results

# Example initialization (can be called from main.py or during app startup)
# initialize_analyzer()