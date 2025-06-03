# new_module3.py
"""
new_module3.py：整合並註冊所有從 brain1.py、brain2.py、brain3.py 搬運來的 EXT_*_Vec 函式。
最終產出 REGISTERED_MODULES_BRAIN dict，以供 analyzer11.py 動態呼叫。
"""
from typing import Dict, Callable
import numpy as np

from brain1 import (
    EXT_A2_Weighted_Proximity_Vec,
    EXT_M3_Local_Heterogeneity_Vec,
    EXT_D3_Potential_Field_Vec,
    EXT_F10_Discontinuity_Vec,
    EXT_P7_Pathfinding_Value_Vec,
    EXT_R5_Resource_Control_Vec,
    EXT_GM1_Row_Control_Vec,
    EXT_GM2_Col_Flow_Vec,
)
from brain2 import (
    EXT_GM3_Adv_Connected_Comp_Vec,
    EXT_GM4_Spatial_Auto_Corr_Vec,
    EXT_GM5_Line_Completion_Vec,
    EXT_GM6_Symmetry_Potential_Vec,
    EXT_GM7_Numeric_Gaps_Vec,
    EXT_GM8_Edge_Affinity_Vec,
    EXT_GM9_Center_Control_Vec,
    EXT_GM10_Blocking_Value_Vec,
    EXT_GM11_Pair_Correlation_Vec,
    EXT_GM12_Island_Analysis_Vec,
)
from brain3 import (
    EXT_GM13_Sequence_Diversity_Vec,
    EXT_GM14_Risk_Assessment_Vec,
    EXT_GM15_Information_Gain_Vec,
    EXT_GM16_Harmonic_Centrality_Vec,
    EXT_GM17_Entropy_Minimization_Vec,
    EXT_GM18_RL_Value_Est_Vec,
    EXT_GM19_Masked_Number_Skip_Pattern_Vec,
    EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec,
)

# 全部註冊至此 dict，方便 analyzer11.py 動態遍歷
REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, str], np.ndarray]] = {
    # …
    "EXT_A2_Weighted_Proximity_Vec": EXT_A2_Weighted_Proximity_Vec,
    "EXT_M3_Local_Heterogeneity_Vec": EXT_M3_Local_Heterogeneity_Vec,
    "EXT_D3_Potential_Field_Vec": EXT_D3_Potential_Field_Vec,
    "EXT_F10_Discontinuity_Vec": EXT_F10_Discontinuity_Vec,
    "EXT_P7_Pathfinding_Value_Vec": EXT_P7_Pathfinding_Value_Vec,
    "EXT_R5_Resource_Control_Vec": EXT_R5_Resource_Control_Vec,
    "EXT_GM1_Row_Control_Vec": EXT_GM1_Row_Control_Vec,
    "EXT_GM2_Col_Flow_Vec": EXT_GM2_Col_Flow_Vec,
    "EXT_GM3_Adv_Connected_Comp_Vec": EXT_GM3_Adv_Connected_Comp_Vec,
    "EXT_GM4_Spatial_Auto_Corr_Vec": EXT_GM4_Spatial_Auto_Corr_Vec,
    "EXT_GM5_Line_Completion_Vec": EXT_GM5_Line_Completion_Vec,
    "EXT_GM6_Symmetry_Potential_Vec": EXT_GM6_Symmetry_Potential_Vec,
    "EXT_GM7_Numeric_Gaps_Vec": EXT_GM7_Numeric_Gaps_Vec,
    "EXT_GM8_Edge_Affinity_Vec": EXT_GM8_Edge_Affinity_Vec,
    "EXT_GM9_Center_Control_Vec": EXT_GM9_Center_Control_Vec,
    "EXT_GM10_Blocking_Value_Vec": EXT_GM10_Blocking_Value_Vec,
    "EXT_GM11_Pair_Correlation_Vec": EXT_GM11_Pair_Correlation_Vec,
    "EXT_GM12_Island_Analysis_Vec": EXT_GM12_Island_Analysis_Vec,
    "EXT_GM13_Sequence_Diversity_Vec": EXT_GM13_Sequence_Diversity_Vec,
    "EXT_GM14_Risk_Assessment_Vec": EXT_GM14_Risk_Assessment_Vec,
    "EXT_GM15_Information_Gain_Vec": EXT_GM15_Information_Gain_Vec,
    "EXT_GM16_Harmonic_Centrality_Vec": EXT_GM16_Harmonic_Centrality_Vec,
    "EXT_GM17_Entropy_Minimization_Vec": EXT_GM17_Entropy_Minimization_Vec,
    "EXT_GM18_RL_Value_Est_Vec": EXT_GM18_RL_Value_Est_Vec,
    "EXT_GM19_Masked_Number_Skip_Pattern_Vec": EXT_GM19_Masked_Number_Skip_Pattern_Vec,
    "EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec": EXT_GM20_Bonus_for_Filling_Internal_Gap_Vec,
}