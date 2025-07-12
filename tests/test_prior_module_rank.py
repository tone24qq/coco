import json
import zipfile

import numpy as np

import analyzer


def _make_samples(path):
    data1 = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    data2 = {"rows": 2, "cols": 2, "grid": [[2, 1], [4, 3]]}
    with zipfile.ZipFile(path / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data1))
        zf.writestr("b.json", json.dumps(data2))


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
    data = {"rows": 2, "cols": 2, "grid": [[1, 2], [3, 4]]}
    with zipfile.ZipFile(samples / "s.zip", "w") as zf:
        zf.writestr("a.json", json.dumps(data))
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


def test_rank_cells_uniform_tie_break(monkeypatch):
    grid = np.full((3, 3), -1)
    cube = np.ones((3, 3, 10), dtype=float)

    monkeypatch.setattr(
        analyzer,
        "get_module_score",
        lambda mod, g, target=None: np.ones_like(g, dtype=float),
    )

    ranks = analyzer.rank_cells_by_prior_and_modules(
        grid,
        cube,
        ["dummy"],
        [1.0],
        target_num=1,
        w_prior=0.5,
    )
    assert all((r, c) != (1, 1) for r, c, _ in ranks)
