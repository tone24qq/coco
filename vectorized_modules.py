"""
vectorized_modules.py - Interface definitions for 4 modules

This module uses lazy initialization for VectorizedBrainModules. If you want to disable
a scoring function, simply comment out the corresponding key in SCORING_MODULES without
modifying other parts.
"""
from vectorized_brain_modules import VectorizedBrainModules

_brain_instance = None

def get_brain():
    """Get the singleton instance of VectorizedBrainModules."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = VectorizedBrainModules()
    return _brain_instance

SCORING_MODULES = {
    'edge_proximity_fusion': lambda grid: get_brain().edge_proximity_fusion(grid),  # Corresponds to edge_proximity_fusion
    'sequence_tail_analyzer': lambda grid: get_brain().sequence_tail_analyzer(grid),  # Corresponds to sequence_tail_analyzer
    'connectivity_heatmap': lambda grid: get_brain().connectivity_heatmap(grid),  # Corresponds to connectivity_heatmap
    'entropy_risk_fusion': lambda grid: get_brain().entropy_risk_fusion(grid),  # Corresponds to entropy_risk_fusion
    'detect_skip_patterns': lambda grid: get_brain().detect_skip_patterns(grid),  # Corresponds to detect_skip_patterns
    'compute_focus_score': lambda grid: get_brain().compute_focus_score(grid),  # Corresponds to compute_focus_score
    'detect_mirror_sequences': lambda grid: get_brain().detect_mirror_sequences(grid),  # Corresponds to detect_mirror_sequences
    'compute_difference_trend': lambda grid: get_brain().compute_difference_trend(grid),  # Corresponds to compute_difference_trend
}