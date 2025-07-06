import numpy as np

import analyzer


def test_predict_uses_simulation_when_neighbors_sparse(monkeypatch):
    grid = [
        [-1, -1],
        [-1, 4],
    ]
    called = {"n": 0}

    def fake_heatmap(g, k, n_iter=500, **_):
        called["n"] += 1
        called["iter"] = n_iter
        return np.zeros_like(g, dtype=float)

    monkeypatch.setattr(analyzer, "probability_heatmap", fake_heatmap)
    result = analyzer.predict_scratch_card(grid, target_num=1, iterations=4)
    assert called["n"] == 1
    assert called.get("iter") == 10000
    assert result.get("strategy") == "heatmap_only"


def test_heatmap_only_when_target_isolated(monkeypatch):
    grid = [
        [5, -1, 1],
        [-1, -1, 2],
        [3, -1, 4],
    ]
    called = {"n": 0}

    def fake_heatmap(g, k, n_iter=500, **_):
        called["n"] += 1
        called["iter"] = n_iter
        return np.zeros_like(g, dtype=float)

    monkeypatch.setattr(analyzer, "probability_heatmap", fake_heatmap)
    result = analyzer.predict_scratch_card(grid, target_num=5, iterations=4)
    assert called["n"] == 1
    assert called.get("iter") == 10000
    assert result.get("strategy") == "heatmap_only"


def test_target_with_diagonal_neighbor_not_heatmap_only(monkeypatch):
    grid = [
        [8, -1, -1],
        [-1, 5, -1],
        [-1, -1, -1],
    ]
    called = {"n": 0}

    def fake_heatmap(g, k, n_iter=500, **_):
        called["n"] += 1
        called["iter"] = n_iter
        return np.zeros_like(g, dtype=float)

    monkeypatch.setattr(analyzer, "probability_heatmap", fake_heatmap)
    result = analyzer.predict_scratch_card(grid, target_num=5, iterations=4)
    assert result.get("strategy") != "heatmap_only"
    if called["n"]:
        assert called.get("iter") != 10000
