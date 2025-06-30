import numpy as np

import brain


def test_row_col_bias_shape_and_range():
    grid = np.array([[1, -1], [2, -1]])
    s = brain.EXT_Q17_RowColBias_Vec(grid)
    assert s.shape == grid.shape
    assert np.all(s[grid != -1] == 0)
    assert np.all(s >= 0)
    assert float(s.max()) <= 1.0
