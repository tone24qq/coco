import numpy as np

from analyzer import simulate_full_board


def test_simulate_with_mask():
    grid = np.array([[1, 2], [3, 4]])
    mask = np.array([[False, True], [False, True]])
    probs = simulate_full_board(grid, None, n_iter=10, mask=mask)
    assert set(probs.keys()) == {(0, 1), (1, 1)}
    for p in probs.values():
        for v in p.values():
            assert 0.0 <= v <= 1.0
