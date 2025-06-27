import numpy as np

import analyzer
import brain


def test_m11_module_shape_and_range():
    grid = np.array([[1, 2], [2, -1]])
    s = brain.EXT_M11_Mirror_Sequence_Vec(grid)
    assert s.shape == grid.shape
    assert np.all(np.isfinite(s))


def test_m11_detects_sequential_pairs():
    grid = np.array([[1, 8], [7, 2]])
    s = brain.EXT_M11_Mirror_Sequence_Vec(grid)
    assert s[0, 0] > 0
    assert s[1, 1] > 0
    assert s[0, 1] > 0
    assert s[1, 0] > 0


def test_select_modules_includes_m11():
    grid = np.array([[1, -1], [2, 3]])
    mods = analyzer.select_modules(grid, target=None)
    assert "EXT_M11_Mirror_Sequence_Vec" in mods


def test_m11_weight_negative():
    assert brain.AGG_WEIGHTS["EXT_M11_Mirror_Sequence_Vec"] < 0
