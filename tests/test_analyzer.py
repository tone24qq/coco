# tests/test_analyzer.py
import numpy as np
from analyzer import simulate_full_board

def test_simulate_dimensions(make_grid):
    grid = np.array(make_grid(6, 7))
    probs = simulate_full_board(grid, iterations=64)
    assert probs.shape == grid.shape
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)

def test_simulate_runs_on_min_board(make_grid):
    # 防止 k-means 因 cluster>K 而掛掉
    grid = np.array(make_grid(2, 2))
    simulate_full_board(grid, iterations=8)  # 只要不拋錯就過