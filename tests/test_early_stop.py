import numpy as np

from analyzer import simulate_full_board


def test_early_stop_consistency(make_grid):
    grid = np.array(make_grid(4, 4))
    rng = np.random.default_rng(0)
    full = simulate_full_board(grid, None, n_iter=20, rng=rng, threshold=0.0)
    rng = np.random.default_rng(0)
    early = simulate_full_board(grid, None, n_iter=20, rng=rng, threshold=1.0)
    assert full.keys() == early.keys()
    for k in full:
        nums = set(full[k]) | set(early[k])
        for n in nums:
            assert 0.0 <= early[k].get(n, 0.0) <= 1.0
