import numpy as np
import pytest
from vectorized_brain_modules import VectorizedBrainModules

def test_edge_proximity_fusion():
    brain = VectorizedBrainModules()
    grid = np.array([[1, -1, 2], [-1, 3, -1], [4, -1, 5]])
    scores = brain.edge_proximity_fusion(grid)
    assert np.all(scores >= 0) and np.all(scores <= 1), "Scores should be normalized between 0 and 1"

def test_sequence_tail_analyzer():
    brain = VectorizedBrainModules()
    grid = np.array([[1, 2, -1], [4, 5, 6], [7, 8, 9]])
    scores = brain.sequence_tail_analyzer(grid)
    assert np.any(scores > 0), "Should detect some sequence tails"

def test_connectivity_heatmap():
    brain = VectorizedBrainModules()
    grid = np.array([[1, -1, 2], [-1, 3, -1], [4, -1, 5]])
    scores = brain.connectivity_heatmap(grid)
    assert np.all(scores >= 0) and np.all(scores <= 1), "Scores should be normalized"

def test_entropy_risk_fusion():
    brain = VectorizedBrainModules()
    grid = np.array([[1, 2, 3], [4, -1, 5], [6, 7, 8]])
    scores = brain.entropy_risk_fusion(grid)
    assert np.any(scores > 0), "Should detect some entropy risk"

def test_detect_skip_patterns():
    brain = VectorizedBrainModules()
    grid = np.array([[1, -1, 3], [-1, 5, -1], [7, -1, 9]])
    scores = brain.detect_skip_patterns(grid)
    assert np.any(scores == 0.9), "Should detect skip pattern"

def test_compute_focus_score():
    brain = VectorizedBrainModules()
    grid = np.array([[1, 2, 3], [4, -1, 5], [6, 7, 8]])
    scores = brain.compute_focus_score(grid)
    assert scores[1, 1] > 0, "Center should have focus score"

def test_detect_mirror_sequences():
    brain = VectorizedBrainModules()
    grid = np.array([[3, 4, -1], [-1, -1, -1], [-1, -1, -1]])
    scores = brain.detect_mirror_sequences(grid)
    assert scores[0, 2] == 0.8, "Should detect mirror sequence"

def test_compute_difference_trend():
    brain = VectorizedBrainModules()
    grid = np.array([[1, 2, -1], [4, 5, 6], [7, 8, 9]])
    scores = brain.compute_difference_trend(grid)
    assert scores[0, 2] > 0, "Should detect difference trend"

def test_test_with_masking():
    brain = VectorizedBrainModules()
    grid = np.arange(1, 10).reshape(3, 3)
    grid[1, 1] = 7
    mean_acc, std_acc = brain.test_with_masking(grid, n_mask=2, target=7, n_trials=5)
    assert 0 <= mean_acc <= 1, "Accuracy should be between 0 and 1"