import numpy as np

import analyzer
import brain
import modules


def test_mirror_module_shape_and_range():
    grid = np.array([[1, 2], [2, -1]])
    s = modules.detect_mirror_sequences(grid)
    assert s.shape == grid.shape
    assert np.all(np.isfinite(s))


def test_mirror_detects_pairs():
    grid = np.array([[1, 8], [7, 2]])
    s = modules.detect_mirror_sequences(grid)
    assert s.shape == grid.shape
    assert np.all(s >= 0)


def test_select_modules_includes_mirror():
    grid = np.array([[1, -1], [2, 3]])
    mods = analyzer.select_modules(grid, target=None)
    assert "mirror" in mods


def test_weight_positive():
    assert brain.AGG_WEIGHTS["mirror"] > 0
