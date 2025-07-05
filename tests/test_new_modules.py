import numpy as np

import analyzer
import brain


def test_modules_registered():
    names = [
        "focus",
        "skip",
        "diff",
        "mirror",
        "conn",
        "tail",
        "gdiff",
        "affinity",
        "gradient_affinity",
        "row_col_bias",
        "row_col_frequency_score",
        "entropy_spread_score",
    ]
    for name in names:
        assert name in brain.REGISTERED_MODULES_BRAIN
        assert name in brain.AGG_WEIGHTS
    assert "modern" in brain.REGISTERED_MODULES


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
    for name in ["focus", "skip", "diff", "mirror", "tail", "gdiff"]:
        assert name in mods


def test_gdiff_weight_positive():
    assert brain.AGG_WEIGHTS["gdiff"] > 0


def test_gdiff_in_legacy_scores():
    grid = [[1, 2, 3], [4, -1, 6], [7, 8, 9]]
    result = analyzer.predict_scratch_card(
        grid,
        iterations=0,
        global_iter=1,
        focus_iter=0,
        unique=False,
    )
    assert result["predictions"]
