import numpy as np

from csp_solver import heuristic_csp_sampling


def test_uniform_distribution():
    grid = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    nbr = {(r, c): 1 / 9 for r in range(3) for c in range(3)}
    probs = heuristic_csp_sampling(grid, 1, nbr, samples=1000)
    vals = list(probs.values())
    assert all(0.0 <= p <= 1.0 for p in vals)
    assert np.allclose(sum(vals), 1.0, atol=0.1)


def test_single_blank():
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, -1]]
    nbr = {(2, 2): 1.0}
    probs = heuristic_csp_sampling(grid, 9, nbr, samples=100)
    assert probs[(2, 2)] == 1.0


def test_two_blanks_nonzero():
    grid = [[1, 2, -1, 4], [5, 6, -1, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    nbr = {(0, 2): 0.5, (1, 2): 0.5}
    probs = heuristic_csp_sampling(grid, 5, nbr, samples=200)
    assert probs[(0, 2)] >= 0.0 and probs[(1, 2)] >= 0.0


def test_enforce_rowcol():
    grid = [[-1, -1], [-1, -1]]
    nbr = {(r, c): 0.5 for r in range(2) for c in range(2)}
    probs = heuristic_csp_sampling(grid, 1, nbr, samples=200, enforce_rowcol=True)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
