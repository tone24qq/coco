import time

import numpy as np

from analyzer import simulate_full_board


def test_simulate_full_board_time_limit():
    grid = np.array([[-1, 1], [2, 3]])
    start = time.monotonic()
    simulate_full_board(grid, None, n_iter=100000, time_limit=0.0)
    assert time.monotonic() - start < 0.2
