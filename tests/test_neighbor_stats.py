import numpy as np

from neighbor_stats import (compute_neighbor_distribution,
                            neighbor_compatibility_score)


def test_compute_neighbor_distribution_sum():
    dist = compute_neighbor_distribution(2, 2, target=1, n_sims=100)
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_neighbor_compatibility_score_shape():
    grid = np.array([[1, -1], [3, -1]])
    dist = compute_neighbor_distribution(2, 2, target=1, n_sims=50)
    score = neighbor_compatibility_score(grid, dist)
    assert score.shape == grid.shape
    assert np.isfinite(score).all()
    if score.max() > 0:
        assert np.isclose(score.max(), 1.0)
