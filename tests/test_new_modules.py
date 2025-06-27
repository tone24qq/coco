import numpy as np

import analyzer
import brain


def test_new_modules_registered():
    for name in [
        "EXT_M12_RestoreOriginalValue_Vec",
        "EXT_Q14_TargetAffinity_Vec",
        "EXT_Q15_GlobalSpread_Vec",
    ]:
        assert name in brain.REGISTERED_MODULES_BRAIN
        assert name in brain.AGG_WEIGHTS


def test_new_module_shapes(make_grid):
    grid = np.array(make_grid(4, 4))
    original = np.array(grid)
    grid[1, 1] = -1
    s1 = brain.EXT_M12_RestoreOriginalValue_Vec(grid, original_grid=original)
    s2 = brain.EXT_Q14_TargetAffinity_Vec(grid, target=3)
    s3 = brain.EXT_Q15_GlobalSpread_Vec(grid)
    for s in (s1, s2, s3):
        assert s.shape == grid.shape
        assert np.isfinite(s).all()


def test_select_modules_include_new(make_grid):
    grid = np.array(make_grid(4, 4))
    mods = analyzer.select_modules(grid, target=1)
    for name in [
        "EXT_M12_RestoreOriginalValue_Vec",
        "EXT_Q14_TargetAffinity_Vec",
        "EXT_Q15_GlobalSpread_Vec",
    ]:
        assert name in mods
