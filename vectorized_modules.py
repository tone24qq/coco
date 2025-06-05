"""
vectorized_modules.py - Interface definitions for 4 modules
"""
from vectorized_brain_modules import VectorizedBrainModules

brain = VectorizedBrainModules()

SCORING_MODULES = {
    'edge_proximity_fusion': brain.edge_proximity_fusion,
    'sequence_tail_analyzer': brain.sequence_tail_analyzer,
    'connectivity_heatmap': brain.connectivity_heatmap,
    'entropy_risk_fusion': brain.entropy_risk_fusion
}