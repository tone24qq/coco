import numpy as np

from analyzer import mc_fullboard_fast
from csp_solver import csp_with_hint


def test_mc_fullboard_fast_single_blank():
    grid = [[1, 2], [3, -1]]
    probs = mc_fullboard_fast(grid, 4, n_iter=10, sample_prob=0.0)
    assert probs[(1, 1)] == 1.0


def test_csp_with_hint_single_blank():
    grid = [[1, 2], [3, -1]]
    probs = csp_with_hint(grid, 4, max_solutions=10, time_limit=0.1)
    assert probs[(1, 1)] == 1.0
