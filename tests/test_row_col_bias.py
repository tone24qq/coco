import numpy as np

import modules


def test_focus_score_range():
    grid = np.array([[1, -1], [2, -1]])
    s = modules.compute_focus_score(grid)
    assert s.shape == grid.shape
    assert np.all(s >= 0)
    assert float(s.max()) <= 1.0
