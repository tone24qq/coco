# analyzer.py

import numpy as np
from typing import Dict, Any
from collections import OrderedDict

from pydantic import BaseModel

from new_module import PuzzleTensorOps
from brain1 import BaseModuleConfig as GM1_3_Config
from brain2 import BaseModuleConfig as GM4_12_Config
from brain3 import BaseModuleConfig as GM13_26_Config

# 导入所有 GM 模块函数
from brain1 import (
    EXT_GM1_Proximity_Vec,
    EXT_GM2_Heterogeneity_Vec,
    EXT_GM3_PotentialField_Vec,
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
)


class CombinedScoresResponse(BaseModel):
    combined_score_matrix: list[list[float]]
    individual_module_scores: dict[str, list[list[float]]]


def compute_combined_scores(
    grid: np.ndarray,
    module_configs: Dict[str, Any],
    request_id: str,
) -> Dict[str, Any]:
    """
    使用 PuzzleTensorOps.score_full_board 一次性全图打分，并保留 individual_scores。
    - grid: 二维 numpy 数组, 空格标记为 -1
    - module_configs: {
          'GM1': GM1_3_Config(...),
          ...,
          'GM26': GM13_26_Config(...)
      }
    - request_id: 用于日志追踪
    """
    rows, cols = grid.shape

    # 构造 GM1–GM26 的配置
    # 若缺失配置，则使用默认：enabled=True, weight=1.0
    default_cfg_1_3 = GM1_3_Config(enabled=True, weight=1.0)
    default_cfg_4_12 = GM4_12_Config(enabled=True, weight=1.0)
    default_cfg_13_26 = GM13_26_Config(enabled=True, weight=1.0)

    full_configs = {
        "GM1": module_configs.get("GM1", default_cfg_1_3),
        "GM2": module_configs.get("GM2", default_cfg_1_3),
        "GM3": module_configs.get("GM3", default_cfg_1_3),
        "GM4": module_configs.get("GM4", default_cfg_4_12),
        "GM5": module_configs.get("GM5", default_cfg_4_12),
        "GM6": module_configs.get("GM6", default_cfg_4_12),
        "GM7": module_configs.get("GM7", default_cfg_4_12),
        "GM8": module_configs.get("GM8", default_cfg_4_12),
        "GM9": module_configs.get("GM9", default_cfg_4_12),
        "GM10": module_configs.get("GM10", default_cfg_4_12),
        "GM11": module_configs.get("GM11", default_cfg_4_12),
        "GM12": module_configs.get("GM12", default_cfg_4_12),
        "GM13": module_configs.get("GM13", default_cfg_13_26),
        "GM14": module_configs.get("GM14", default_cfg_13_26),
        "GM15": module_configs.get("GM15", default_cfg_13_26),
        "GM16": module_configs.get("GM16", default_cfg_13_26),
        "GM17": module_configs.get("GM17", default_cfg_13_26),
        "GM18": module_configs.get("GM18", default_cfg_13_26),
        "GM19": module_configs.get("GM19", default_cfg_13_26),
        "GM20": module_configs.get("GM20", default_cfg_13_26),
        "GM21": module_configs.get("GM21", default_cfg_13_26),
        "GM22": module_configs.get("GM22", default_cfg_13_26),
        "GM23": module_configs.get("GM23", default_cfg_13_26),
        "GM24": module_configs.get("GM24", default_cfg_13_26),
        "GM25": module_configs.get("GM25", default_cfg_13_26),
        "GM26": module_configs.get("GM26", default_cfg_13_26),
    }

    # 1. 使用 score_full_board 一次性计算合并分数
    pto = PuzzleTensorOps(grid)
    combined_scores = pto.score_full_board(full_configs, request_id)

    # 2. 依旧生成 individual_scores，以便调试或返回给前端
    individual_scores = OrderedDict()
    modules = [
        ("GM1", EXT_GM1_Proximity_Vec),
        ("GM2", EXT_GM2_Heterogeneity_Vec),
        ("GM3", EXT_GM3_PotentialField_Vec),
        ("GM4", EXT_GM4_Spatial_Auto_Corr_Vec),
        ("GM5", EXT_GM5_Line_Completion_Vec),
        ("GM6", EXT_GM6_Symmetry_Potential_Vec),
        ("GM7", EXT_GM7_Numeric_Gaps_Vec),
        ("GM8", EXT_GM8_Edge_Affinity_Vec),
        ("GM9", EXT_GM9_Center_Control_Vec),
        ("GM10", EXT_GM10_BlockingValue_Vec),
        ("GM11", EXT_GM11_PairCorrelation_Vec),
        ("GM12", EXT_GM12_IslandAnalysis_Vec),
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
    ]

    for name, func in modules:
        cfg = full_configs[name]
        try:
            scores = func(grid, cfg, request_id)
            if not isinstance(scores, np.ndarray) or scores.shape != (rows, cols):
                raise ValueError(f"{name} 返回形状错误: {scores.shape}")
        except Exception:
            scores = np.zeros((rows, cols), dtype=float)
        individual_scores[name] = scores.tolist()

    response = CombinedScoresResponse(
        combined_score_matrix=combined_scores.tolist(),
        individual_module_scores=individual_scores,
    )
    return response.dict()