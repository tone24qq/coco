import numpy as np
import analyzer


def test_neighbor_lock_hits(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])

    def fake_score(g, dist):
        s = np.zeros_like(g, dtype=float)
        s[0, 1] = 0.8
        return s

    monkeypatch.setattr(analyzer, "neighbor_compatibility_score", fake_score)
    monkeypatch.setattr(analyzer, "compute_neighbor_distribution", lambda *a, **k: {})
    called = {"sim": 0}

    def fake_sim(*a, **k):
        called["sim"] += 1
        return {}

    monkeypatch.setattr(analyzer, "simulate_full_board", fake_sim)

    res = analyzer.predict_scratch_card(
        grid.tolist(),
        target_num=1,
        iterations=4,
        use_neighbor_lock=True,
        sample_gamma=0.0,
        fusion_alpha=0.0,
    )
    # allow either pure neighbor or fallback
    assert res["strategy"] in ("neighbor_lock", "pure_sample+global")
    assert called["sim"] == 0
    assert res["predictions"][0]["row"] == 0
    assert res["predictions"][0]["col"] == 1


def test_neighbor_lock_fallback(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])

    monkeypatch.setattr(
        analyzer,
        "neighbor_compatibility_score",
        lambda g, d: np.zeros_like(g, dtype=float),
    )
    monkeypatch.setattr(analyzer, "compute_neighbor_distribution", lambda *a, **k: {})
    called = {"sim": 0, "prior": 0}

    def fake_sim(g, t, n_iter=0, **_):
        called["sim"] += 1
        return {(0, 1): {1: 0.4}}

    def fake_prior(*a, **k):
        called["prior"] += 1
        return {(0, 1): {1: 0.6}}

    monkeypatch.setattr(analyzer, "simulate_full_board", fake_sim)
    monkeypatch.setattr(analyzer, "compute_position_probabilities", fake_prior)

    res = analyzer.predict_scratch_card(
        grid.tolist(),
        target_num=1,
        iterations=2,
        use_neighbor_lock=True,
        sample_gamma=0.0,
        fusion_alpha=0.0,
    )
    assert res["strategy"] in ("neighbor_lock", "pure_sample+global")
    assert called["sim"] == 1
    assert called["prior"] == 1
    assert res["predictions"][0]["col"] == 1