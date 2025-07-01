import numpy as np

import modules


def test_row_col_bias_range():
    grid = np.array([[1, -1], [2, -1]])
    s = modules.row_col_bias(grid)
    assert s.shape == grid.shape
    assert np.all(s >= 0)
    assert float(s.max()) <= 1.0
