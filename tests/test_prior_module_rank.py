import numpy as np

import analyzer


def test_compute_global_distribution():
    cube = analyzer.compute_global_distribution("", 2, 2, n_synth=20, seed=0)
    assert cube.shape == (2, 2, 5)
    for r in range(2):
        for c in range(2):
            total = cube[r, c, 1:].sum()
            assert abs(total - 1.0) < 1e-6


def test_rank_cells_by_prior_and_modules():
    cube = analyzer.compute_global_distribution("", 2, 2, n_synth=20, seed=0)
    grid = np.array([[-1, 2], [3, -1]])
    mods = ["EXT_Q1_ProximityEntropy_Vec"]
    ranks = analyzer.rank_cells_by_prior_and_modules(
        grid,
        cube,
        mods,
        [1.0],
        target_num=1,
        w_prior=1.0,
    )
    assert ranks[0][:2] == (0, 0)
    assert len(ranks) == 2
    assert all(grid[r, c] == -1 for r, c, _ in ranks)


def test_rank_cells_normalization(monkeypatch):
    cube = analyzer.compute_global_distribution("", 2, 2, n_synth=20, seed=0)

    grid = np.array([[-1, -1], [3, -1]])

    monkeypatch.setattr(
        analyzer,
        "get_module_score",
        lambda mod, g, target=None: np.ones_like(g, dtype=float),
    )

    ranks = analyzer.rank_cells_by_prior_and_modules(
        grid,
        cube,
        ["DUMMY"],
        [1.0],
        target_num=1,
        w_prior=0.5,
    )

    total_prob = sum(p / 100.0 for _, _, p in ranks)
    assert abs(total_prob - 1.0) < 1e-6
    assert ranks[0][2] < 100.0
    assert ranks[0][2] >= ranks[1][2] >= ranks[2][2]
