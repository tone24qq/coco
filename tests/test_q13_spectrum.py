import numpy as np

import analyzer
import brain
import modules


def random_board(rng, r, c):
    board = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
    blanks = rng.choice(r * c, max(1, (r * c) // 4), replace=False)
    board.ravel()[blanks] = -1
    return board


def test_connectivity_heatmap_basic():
    rng = np.random.default_rng(0)
    for _ in range(3):
        r = int(rng.integers(4, 8))
        c = int(rng.integers(4, 8))
        grid = random_board(rng, r, c)
        s = modules.connectivity_heatmap(grid)
        assert s.shape == grid.shape
        assert np.all(s[grid != -1] >= 0)


def test_conn_weight_low():
    assert brain.AGG_WEIGHTS["conn"] <= 0.05
