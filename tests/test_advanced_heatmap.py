import numpy as np

import analyzer


def test_local_adaptive_heatmap_normalizes(monkeypatch):
    grid = np.array([[1, -1], [2, -1]])
    fake_global = np.array([[0.2, 0.3], [0.3, 0.2]])
    monkeypatch.setattr(analyzer, "get_global_heatmap", lambda *a, **k: fake_global)
    out = analyzer.local_adaptive_heatmap(grid, 1)
    assert out.shape == grid.shape
    assert out[0, 0] == 0.0
    assert np.isclose(out.sum(), 1.0)


def test_conditional_heatmap_uses_matches(monkeypatch):
    grid = np.array([[1, -1], [2, -1]])
    m1 = np.array([[1, 3], [2, 4]])
    m2 = np.array([[1, 4], [2, 3]])
    monkeypatch.setattr(analyzer, "match_samples", lambda *a, **k: [m1, m2])
    monkeypatch.setattr(analyzer, "get_global_heatmap", lambda *a, **k: np.ones((2, 2)))
    out = analyzer.conditional_heatmap(grid, 3)
    assert out[0, 1] > 0 and out[1, 1] > 0
    assert np.isclose(out.sum(), 1.0)


def test_neighbor_weighted_heatmap_boosts_neighbors(monkeypatch):
    grid = np.array([[1, -1], [-1, 2]])
    base = np.array([[0.1, 0.4], [0.3, 0.2]])
    monkeypatch.setattr(analyzer, "local_adaptive_heatmap", lambda *a, **k: base)
    out = analyzer.neighbor_weighted_heatmap(
        grid, 1, weight_by_value=lambda v: 2.0 if v == 1 else 1.0
    )
    assert out.shape == grid.shape
    assert out[0, 1] > out[1, 0]
    assert np.isclose(out.sum(), 1.0)


def test_sample_enhanced_heatmap_fallback(monkeypatch):
    grid = np.array([[1, -1], [2, -1]])
    monkeypatch.setattr(analyzer, "match_samples", lambda *a, **k: [])
    monkeypatch.setattr(
        analyzer,
        "get_global_heatmap",
        lambda *a, **k: np.array([[0.4, 0.1], [0.3, 0.2]]),
    )
    out = analyzer.sample_enhanced_heatmap(grid, 4)
    assert out.shape == grid.shape
    assert np.isclose(out.sum(), 1.0)
