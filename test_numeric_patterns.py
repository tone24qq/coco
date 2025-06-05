import numpy as np
import pytest
from analyzer11_optimized import detect_skip_patterns, compute_focus_score, detect_mirror_sequences, compute_difference_trend

def test_detect_skip_patterns():
    # Test case 1: 3x3 grid with skip pattern
    grid1 = np.array([[1, -1, 3], [-1, 5, -1], [7, -1, 9]])
    heatmap = detect_skip_patterns(grid1)
    assert np.any(heatmap == 0.9), "Should detect skip pattern with score 0.9"
    
    # Test case 2: 4x4 grid with no skip pattern
    grid2 = np.array([[1, 2, 3, 4], [5, -1, 7, 8], [9, 10, 11, -1], [13, 14, 15, 16]])
    heatmap = detect_skip_patterns(grid2)
    assert np.all(heatmap[heatmap > 0] < 0.9), "No strong skip pattern should yield lower scores"

def test_compute_focus_score():
    # Test case 1: 3x3 grid with dense center
    grid1 = np.array([[1, 2, 3], [4, -1, 5], [6, 7, 8]])
    heatmap = compute_focus_score(grid1)
    assert heatmap[1, 1] > 0, "Center blank should have positive focus score"
    
    # Test case 2: 4x4 grid with sparse distribution
    grid2 = np.array([[1, -1, -1, 4], [-1, -1, -1, -1], [-1, -1, -1, -1], [16, -1, -1, -1]])
    heatmap = compute_focus_score(grid2)
    assert heatmap[0, 1] < heatmap[0, 0], "Sparse areas should have lower scores"

def test_detect_mirror_sequences():
    # Test case 1: 3x3 grid with mirror sequence
    grid1 = np.array([[3, 4, -1], [-1, -1, -1], [-1, -1, -1]])
    heatmap = detect_mirror_sequences(grid1)
    assert heatmap[0, 2] == 0.8, "Should detect mirror sequence 3,4,-1 -> 5"
    
    # Test case 2: 4x4 grid with no mirror sequence
    grid2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, -1, 12], [13, 14, 15, 16]])
    heatmap = detect_mirror_sequences(grid2)
    assert np.all(heatmap == 0), "No mirror sequence should yield zero scores"

def test_compute_difference_trend():
    # Test case 1: 3x3 grid with arithmetic trend
    grid1 = np.array([[1, 2, -1], [4, 5, 6], [7, 8, 9]])
    heatmap = compute_difference_trend(grid1)
    assert heatmap[0, 2] > 0, "Should detect trend 1,2,-1 with expected 3"
    
    # Test case 2: 4x4 grid with no clear trend
    grid2 = np.array([[1, 3, 5, 7], [2, -1, 4, 6], [3, 5, -1, 8], [4, 6, 8, 10]])
    heatmap = compute_difference_trend(grid2)
    assert np.all(heatmap < 0.7), "No strong trend should yield lower scores"