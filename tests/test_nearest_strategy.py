import numpy as np

import brain
from analyzer import probability_heatmap
from modules import neighbor_value_distribution


def test_neighbor_value_fallback():
    grid = np.array([[1, -1], [10, 15]])
    base = neighbor_value_distribution(grid, target=50, tolerance=0)
    near = neighbor_value_distribution(grid, target=50, tolerance=0, nearest_k=1)
    assert near[0, 1] > base[0, 1]


def test_probability_heatmap_nearest_weight():
    grid = np.array([[1, -1], [2, -1]])
    hm = probability_heatmap(grid, 2, n_iter=8, seed=1, nearest_weight=0.5)
    assert hm.shape == grid.shape
    assert np.all(np.isfinite(hm))


def test_crf_nearest_value_integration():
    grid = np.array([[1, -1], [2, 3]])
    cooc = {(-1, 0): {1: {2: 1.0}}, (1, 0): {}, (0, -1): {}, (0, 1): {}}
    out = brain.EXT_X_CRFInference(
        grid,
        target=2,
        cooc_prob=cooc,
        nearest_k=1,
        iterations=1,
    )
    assert out.shape == (grid.shape[0], grid.shape[1], grid.size + 1)
    assert np.allclose(out.sum(axis=2)[grid == -1], 1.0, atol=1e-6)
