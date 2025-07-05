# isort: skip_file
import numpy as np

from neighbor_stats import compute_neighbor_distribution, neighbor_compatibility_score


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


def test_ranking_prioritizes_high_rank_neighbors():
    grid = np.array(
        [
            [4, 2, 3],
            [-1, 1, -1],
            [5, 6, 7],
        ]
    )
    dist = {4: 0.6, 1: 0.3, 2: 0.1}
    score = neighbor_compatibility_score(grid, dist)
    assert score[1, 0] > score[1, 2]
