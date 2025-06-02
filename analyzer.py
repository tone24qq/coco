# analyzer.py
# 负责业务逻辑调度：将 grid 传入 brain1/brain2/brain3 三个模块，合并 26 个 GM 模块的评分结果。

import logging
from typing import Dict, Tuple, Any

import numpy as np

from brain1 import (
    EXT_GM1_Proximity_Vec,
    EXT_GM2_Heterogeneity_Vec,
    EXT_GM3_PotentialField_Vec,
    BaseModuleConfig,
)
from brain2 import (
    EXT_GM4_Spatial_Auto_Corr_Vec,
    EXT_GM5_Line_Completion_Vec,
    EXT_GM6_Symmetry_Potential_Vec,
    EXT_GM7_Numeric_Gaps_Vec,
    EXT_GM8_Edge_Affinity_Vec,
    EXT_GM9_Center_Control_Vec,
    EXT_GM10_BlockingValue_Vec,
    EXT_GM11_PairCorrelation_Vec,
    EXT_GM12_IslandAnalysis_Vec,
    SpatialAutocorrelationConfig,
    LineCompletionConfig,
    SymmetryPotentialConfig,
    NumericGapsConfig,
    EdgeAffinityConfig,
    CenterControlConfig,
    BlockingValueConfig,
    PairCorrelationConfig,
    IslandAnalysisConfig,
)
from brain3 import (
    EXT_GM13_Sequence_Diversity_Vec,
    EXT_GM14_Risk_Assessment_Vec,
    EXT_GM15_Information_Gain_Vec,
    EXT_GM16_Harmonic_Centrality_Vec,
    EXT_GM17_Local_Entropy_Vec,
    EXT_GM18_RL_Value_Estimation_Vec,
    EXT_GM19_SkipPattern_Vec,
    EXT_GM20_SkipPattern_Confidence_Vec,
    EXT_GM21_ClusterBalance_Vec,
    EXT_GM22_CoOccurrence_Vec,
    EXT_GM23_MotifDetection_Vec,
    EXT_GM24_TemporalCoherence_Vec,
    EXT_GM25_StrategicDepth_Vec,
    EXT_GM26_ContextualFlexibility_Vec,
    SequenceDiversityConfig,
    RiskAssessmentConfig,
    InformationGainConfig,
    HarmonicCentralityConfig,
    LocalEntropyMinimizationConfig,
    RLValueEstimationConfig,
    SkipPatternConfig,
    SkipPatternConfidenceConfig,
)
from typing import Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODULE_CONFIGS: Dict[str, BaseModuleConfig] = {
    "GM1": BaseModuleConfig(enabled=True, weight=1.0),
    "GM2": BaseModuleConfig(enabled=True, weight=1.0),
    "GM3": BaseModuleConfig(enabled=True, weight=1.0),
    "GM4": SpatialAutocorrelationConfig(enabled=True, weight=1.0),
    "GM5": LineCompletionConfig(enabled=True, weight=1.0),
    "GM6": SymmetryPotentialConfig(enabled=True, weight=1.0),
    "GM7": NumericGapsConfig(enabled=True, weight=1.0),
    "GM8": EdgeAffinityConfig(enabled=True, weight=1.0),
    "GM9": CenterControlConfig(enabled=True, weight=1.0),
    "GM10": BlockingValueConfig(enabled=True, weight=1.0),
    "GM11": PairCorrelationConfig(enabled=True, weight=1.0),
    "GM12": IslandAnalysisConfig(enabled=True, weight=1.0),
    "GM13": SequenceDiversityConfig(enabled=True, weight=1.0),
    "GM14": RiskAssessmentConfig(enabled=True, weight=1.0),
    "GM15": InformationGainConfig(enabled=True, weight=1.0),
    "GM16": HarmonicCentralityConfig(enabled=True, weight=1.0),
    "GM17": LocalEntropyMinimizationConfig(enabled=True, weight=1.0),
    "GM18": RLValueEstimationConfig(enabled=True, weight=1.0),
    "GM19": SkipPatternConfig(enabled=True, weight=1.0),
    "GM20": SkipPatternConfidenceConfig(enabled=True, weight=1.0),
    "GM21": BaseModuleConfig(enabled=True, weight=1.0),
    "GM22": BaseModuleConfig(enabled=True, weight=1.0),
    "GM23": BaseModuleConfig(enabled=True, weight=1.0),
    "GM24": BaseModuleConfig(enabled=True, weight=1.0),
    "GM25": BaseModuleConfig(enabled=True, weight=1.0),
    "GM26": BaseModuleConfig(enabled=True, weight=1.0),
}


def compute_combined_scores(
    grid: np.ndarray, overrides: Dict[str, Any], request_id: str
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    对给定 grid 运行 GM1–GM26 共 26 个向量化模块，返回按 cell 合并后的评分字典：
    { (r,c): {"GM1": 0.23, "GM2": 0.45, … "GM26": 0.10} }
    """
    if grid.ndim != 2 or grid.dtype.kind not in ("i",):
        msg = "Grid must be a 2D NumPy array of ints"
        logger.error(msg, extra={"request_id": request_id})
        raise ValueError(msg)

    rows, cols = grid.shape
    covered_coords = list(zip(*np.where(grid == -1)))

    module_configs = DEFAULT_MODULE_CONFIGS.copy()
    for key, val in overrides.items():
        if key in module_configs and isinstance(val, dict):
            module_configs[key] = module_configs[key].model_copy().model_update(val)

    combined: Dict[Tuple[int, int], Dict[str, float]] = {
        (r, c): {} for (r, c) in covered_coords
    }

    # Brain1 (GM1–GM3)
    if module_configs["GM1"].enabled:
        try:
            scores1 = EXT_GM1_Proximity_Vec(grid, module_configs["GM1"], request_id)
            for (r, c) in covered_coords:
                combined[(r, c)]["GM1"] = float(scores1[r, c])
        except Exception as e:
            logger.error(
                f"GM1 error: {e}", extra={"request_id": request_id}, exc_info=True
            )
            for (r, c) in covered_coords:
                combined[(r, c)]["GM1"] = 0.0

    if module_configs["GM2"].enabled:
        try:
            scores2 = EXT_GM2_Heterogeneity_Vec(grid, module_configs["GM2"], request_id)
            for (r, c) in covered_coords:
                combined[(r, c)]["GM2"] = float(scores2[r, c])
        except Exception as e:
            logger.error(
                f"GM2 error: {e}", extra={"request_id": request_id}, exc_info=True
            )
            for (r, c) in covered_coords:
                combined[(r, c)]["GM2"] = 0.0

    if module_configs["GM3"].enabled:
        try:
            scores3 = EXT_GM3_PotentialField_Vec(grid, module_configs["GM3"], request_id)
            for (r, c) in covered_coords:
                combined[(r, c)]["GM3"] = float(scores3[r, c])
        except Exception as e:
            logger.error(
                f"GM3 error: {e}", extra={"request_id": request_id}, exc_info=True
            )
            for (r, c) in covered_coords:
                combined[(r, c)]["GM3"] = 0.0

    # Brain2 (GM4–GM12)
    for gm_key, func in [
        ("GM4", EXT_GM4_Spatial_Auto_Corr_Vec),
        ("GM5", EXT_GM5_Line_Completion_Vec),
        ("GM6", EXT_GM6_Symmetry_Potential_Vec),
        ("GM7", EXT_GM7_Numeric_Gaps_Vec),
        ("GM8", EXT_GM8_Edge_Affinity_Vec),
        ("GM9", EXT_GM9_Center_Control_Vec),
        ("GM10", EXT_GM10_BlockingValue_Vec),
        ("GM11", EXT_GM11_PairCorrelation_Vec),
        ("GM12", EXT_GM12_IslandAnalysis_Vec),
    ]:
        cfg = module_configs[gm_key]
        if cfg.enabled:
            try:
                vec_scores = func(grid, cfg, request_id)
                for (r, c) in covered_coords:
                    combined[(r, c)][gm_key] = float(vec_scores[r, c])
            except Exception as e:
                logger.error(
                    f"{gm_key} error: {e}",
                    extra={"request_id": request_id},
                    exc_info=True,
                )
                for (r, c) in covered_coords:
                    combined[(r, c)][gm_key] = 0.0
        else:
            for (r, c) in covered_coords:
                combined[(r, c)][gm_key] = 0.0

    # Brain3 (GM13–GM26)
    for gm_key, func in [
        ("GM13", EXT_GM13_Sequence_Diversity_Vec),
        ("GM14", EXT_GM14_Risk_Assessment_Vec),
        ("GM15", EXT_GM15_Information_Gain_Vec),
        ("GM16", EXT_GM16_Harmonic_Centrality_Vec),
        ("GM17", EXT_GM17_Local_Entropy_Vec),
        ("GM18", EXT_GM18_RL_Value_Estimation_Vec),
        ("GM19", EXT_GM19_SkipPattern_Vec),
        ("GM20", EXT_GM20_SkipPattern_Confidence_Vec),
        ("GM21", EXT_GM21_ClusterBalance_Vec),
        ("GM22", EXT_GM22_CoOccurrence_Vec),
        ("GM23", EXT_GM23_MotifDetection_Vec),
        ("GM24", EXT_GM24_TemporalCoherence_Vec),
        ("GM25", EXT_GM25_StrategicDepth_Vec),
        ("GM26", EXT_GM26_ContextualFlexibility_Vec),
    ]:
        cfg = module_configs[gm_key]
        if cfg.enabled:
            try:
                vec_scores = func(grid, cfg, request_id)
                for (r, c) in covered_coords:
                    combined[(r, c)][gm_key] = float(vec_scores[r, c])
            except Exception as e:
                logger.error(
                    f"{gm_key} error: {e}",
                    extra={"request_id": request_id},
                    exc_info=True,
                )
                for (r, c) in covered_coords:
                    combined[(r, c)][gm_key] = 0.0
        else:
            for (r, c) in covered_coords:
                combined[(r, c)][gm_key] = 0.0

    return combined