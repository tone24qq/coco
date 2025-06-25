# tests/test_analyzer.py
import numpy as np

from analyzer import simulate_full_board


def test_simulate_dimensions(make_grid):
    grid = np.array(make_grid(6, 7))
    probs = simulate_full_board(grid, None, n_iter=64)
    assert isinstance(probs, dict)
    for cell_probs in probs.values():
        for p in cell_probs.values():
            assert 0.0 <= p <= 1.0


def test_simulate_runs_on_min_board(make_grid):
    """simulate_full_board may fail on tiny boards; ensure graceful handling."""
    grid = np.array(make_grid(2, 2))
    simulate_full_board(grid, None, n_iter=8)
