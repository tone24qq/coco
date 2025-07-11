# tests/test_analyzer.py
import numpy as np
import pytest

import analyzer
from analyzer import simulate_full_board


def test_simulate_dimensions(make_grid):
    grid = np.array(make_grid(6, 7))
    probs = simulate_full_board(grid, None, n_iter=64)
    assert isinstance(probs, dict)
    for cell_probs in probs.values():
        for p in cell_probs.values():
            assert 0.0 <= p <= 1.0


def test_simulate_runs_on_min_board(make_grid):
    """simulate_full_board may fail on tiny boards; ensure graceful handling."""
    grid = np.array(make_grid(2, 2))
    simulate_full_board(grid, None, n_iter=8)


def test_simulate_focus_cells(make_grid):
    grid = np.array(make_grid(4, 4))
    blanks = [tuple(p) for p in np.argwhere(np.array(grid) == -1)]
    probs = simulate_full_board(
        grid,
        None,
        n_iter=4,
        focus_cells=blanks[:1],
        epsilon=0.1,
    )
    assert isinstance(probs, dict)


def test_simulate_target_mode(make_grid):
    grid = np.array(make_grid(3, 3))
    result = simulate_full_board(grid, 5, n_iter=10)
    assert isinstance(result, dict)
    for cell_probs in result.values():
        for p in cell_probs.values():
            assert 0.0 <= p <= 1.0


def test_weight_prob_by_modules_variation(monkeypatch):
    grid = np.array([[1, -1], [3, -1]])
    prob_map = {(0, 1): {2: 0.6}, (1, 1): {4: 0.4}}

    def fake_select_modules(_grid, target=None):
        return ["A", "B"]

    def fake_get_module_score(mod, _grid, target=None):
        arr = np.zeros_like(_grid, dtype=float)
        if mod == "A":
            arr.fill(1.0)
        else:
            arr.fill(0.1)
        return arr

    monkeypatch.setattr(analyzer, "select_modules", fake_select_modules)
    monkeypatch.setattr(analyzer, "get_module_score", fake_get_module_score)
    monkeypatch.setattr(
        analyzer, "Parallel", lambda n_jobs=1: (lambda tasks: [t() for t in tasks])
    )
    monkeypatch.setattr(
        analyzer, "delayed", lambda fn: (lambda *a, **k: lambda: fn(*a, **k))
    )

    weighted = analyzer.weight_prob_by_modules(grid, prob_map)
    assert weighted[(0, 1)][2] != prob_map[(0, 1)][2]


def test_assign_unique_numbers_simple():
    prob_map = {
        (0, 0): {1: 0.8, 2: 0.2},
        (0, 1): {1: 0.1, 2: 0.7, 3: 0.8},
        (1, 0): {1: 0.1, 2: 0.1, 3: 0.2},
    }
    mapping = analyzer.assign_unique_numbers(prob_map)
    assert set(mapping.keys()) == {1, 2, 3}
    assert len(set(mapping.values())) == len(mapping)


def test_probabilities_not_uniform():
    grid = [[1, 2], [3, -1]]
    prob_map = simulate_full_board(grid, None, n_iter=20)
    cell = prob_map[(1, 1)]
    assert len(cell) == 1 and next(iter(cell.values())) == 1.0


def test_predict_always_excludes_filled_cells():
    grid = [[1, -1], [-1, 4]]
    res = analyzer.predict_scratch_card(
        grid,
        target_num=3,
        iterations=4,
        exclude_filled=False,
    )
    coords = {(p["row"], p["col"]) for p in res["predictions"]}
    assert (0, 0) not in coords
    assert (1, 1) not in coords


def test_predict_passes_sample_gamma(monkeypatch):
    captured = {}

    def fake_heatmap(*args, sample_gamma=0.0, history_dir="samples", **kwargs):
        captured["gamma"] = sample_gamma
        captured["history_dir"] = history_dir
        grid = np.asarray(args[0])
        return np.zeros_like(grid, dtype=float)

    monkeypatch.setattr(analyzer, "probability_heatmap", fake_heatmap)

    grid = [[-1, -1], [-1, -1]]
    analyzer.predict_scratch_card(
        grid,
        target_num=1,
        iterations=2,
        global_iter=1,
        focus_iter=1,
        epsilon=1.0,
        sample_gamma=0.7,
        history_dir="foo",
    )

    assert captured["gamma"] == 0.7
    assert captured["history_dir"] == "foo"


def test_apply_uniqueness_penalty():
    pm = {
        (0, 0): {1: 0.6, 2: 0.4},
        (0, 1): {1: 0.5},
    }
    out = analyzer.apply_uniqueness_penalty(pm, strength=1.0)
    assert out[(0, 0)][1] < pm[(0, 0)][1]
    assert out[(0, 0)][2] < pm[(0, 0)][2]
    assert out[(0, 1)][1] == pm[(0, 1)][1]


def test_apply_consecutive_penalty_map():
    pm = {(0, 0): {1: 0.6, 3: 0.4}}
    penalties = {2: 0.5}
    out = analyzer.apply_consecutive_penalty_map(pm, 1, penalties)
    assert out[(0, 0)][3] == pytest.approx(0.4 * 0.5)
