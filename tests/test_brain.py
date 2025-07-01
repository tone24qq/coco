# tests/test_brain.py
import numpy as np

import brain

MODULE_FNS = list(brain.REGISTERED_MODULES_BRAIN.values())


def test_module_shapes(make_grid):
    grid = np.array(make_grid(8, 10))
    for fn in MODULE_FNS:
        out = fn(grid)
        assert out.shape == grid.shape
        assert np.isfinite(out).all()


def test_aggregate_scores_basic():
    stack = np.array(
        [
            np.ones((2, 2)),
            np.full((2, 2), 2.0),
        ]
    )
    weights = np.array([0.7, 0.3])
    out = brain.aggregate_scores(stack, weights, ["A", "B"])
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))


def test_compute_nearest_value_heatmap_basic():
    grid = np.array([[1, -1], [2, 3]])
    cooc = {(-1, 1): {2: {2: 1.0}}}
    heat = brain.compute_nearest_value_heatmap(grid, target=2, cooc_prob=cooc, k=1)
    assert heat.shape == grid.shape
    assert np.all(heat >= 0)
