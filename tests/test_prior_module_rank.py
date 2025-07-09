import numpy as np

import analyzer


def _make_samples(path):
    freq = np.zeros((2, 2, 5), dtype=int)
    boards = [np.array([[1, 2], [3, 4]]), np.array([[2, 1], [4, 3]])]
    for b in boards:
        rr, cc = np.indices(b.shape)
        np.add.at(freq, (rr, cc, b), 1)
    np.savez(
        path / "2x2.npz",
        freq=freq,
        meta={"samples": len(boards), "schema_version": 1, "generated_at": "now"},
    )


def test_compute_global_distribution(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    _make_samples(samples)
    cube = analyzer.compute_global_distribution(str(samples), 2, 2)
    assert cube.shape == (2, 2, 5)
    assert abs(cube[0, 0, 1] - 0.5) < 1e-6
    assert cube[1, 1, 1] == 0.0
    for r in range(2):
        for c in range(2):
            total = cube[r, c, 1:].sum()
            assert abs(total - 1.0) < 1e-6


def test_rank_cells_by_prior_and_modules(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    _make_samples(samples)
    cube = analyzer.compute_global_distribution(str(samples), 2, 2)
    grid = np.array([[-1, 2], [3, -1]])
    mods = ["focus"]
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


def test_rank_cells_normalization(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    freq = np.zeros((2, 2, 5), dtype=int)
    board = np.array([[1, 2], [3, 4]])
    rr, cc = np.indices(board.shape)
    np.add.at(freq, (rr, cc, board), 1)
    np.savez(
        samples / "2x2.npz",
        freq=freq,
        meta={"samples": 1, "schema_version": 1, "generated_at": "now"},
    )
    cube = analyzer.compute_global_distribution(str(samples), 2, 2)

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
