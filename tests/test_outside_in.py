import numpy as np

from analyzer import compute_rings, predict_outside_in


def test_compute_rings_simple():
    grid = np.zeros((3, 3), dtype=int)
    rings = compute_rings(grid)
    assert rings[(0, 0)] == 0
    assert rings[(1, 1)] == 1


def test_predict_outside_in_single_blank():
    grid = [[1, 2], [3, -1]]
    probs = predict_outside_in(
        grid, 4, "samples", mc_iter=10, csp_sols=5, time_limit=0.1
    )
    assert probs[(1, 1)] == 1.0
