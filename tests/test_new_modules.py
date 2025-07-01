import numpy as np

import analyzer
import brain


def test_modules_registered():
    for name in ["focus", "skip", "diff", "mirror", "conn", "tail"]:
        assert name in brain.REGISTERED_MODULES_BRAIN
        assert name in brain.AGG_WEIGHTS


def test_module_shapes(make_grid):
    grid = np.array(make_grid(4, 4))
    grid[1, 1] = -1
    for name in brain.REGISTERED_MODULES_BRAIN:
        s = brain.get_module_score(name, grid)
        assert s.shape == grid.shape
        assert np.isfinite(s).all()


def test_select_modules_include_all(make_grid):
    grid = np.array(make_grid(4, 4))
    mods = analyzer.select_modules(grid, target=1)
    for name in ["focus", "skip", "diff", "mirror", "conn", "tail"]:
        assert name in mods
