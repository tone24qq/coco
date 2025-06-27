import numpy as np

import analyzer
import brain


def test_m11_module_shape_and_range():
    grid = np.array([[1, 2], [2, -1]])
    s = brain.EXT_M11_Mirror_Sequence_Vec(grid)
    assert s.shape == grid.shape
    assert np.all(np.isfinite(s))
    assert np.all(s[grid != -1] == 0)


def test_select_modules_includes_m11():
    grid = np.array([[1, -1], [2, 3]])
    mods = analyzer.select_modules(grid, target=None)
    assert "EXT_M11_Mirror_Sequence_Vec" in mods
