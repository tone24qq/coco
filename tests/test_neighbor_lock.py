import numpy as np
import analyzer


def test_neighbor_lock_hits(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])

    def fake_score(g, dist):
        s = np.zeros_like(g, dtype=float)
        s[0, 1] = 0.8  # 模擬鄰居分數推 0,1 是最佳格
        return s

    monkeypatch.setattr(analyzer, "neighbor_compatibility_score", fake_score)
    monkeypatch.setattr(analyzer, "compute_neighbor_distribution", lambda *a, **k: {})
    called = {"sim": 0}

    def fake_sim(*a, **k):
        called["sim"] += 1
        return {}  # 不應被呼叫，模擬分數不會用到

    monkeypatch.setattr(analyzer, "simulate_full_board", fake_sim)

    res = analyzer.predict_scratch_card(
        grid.tolist(),
        target_num=1,
        iterations=4,
        use_neighbor_lock=True,
        sample_gamma=0.0,  # 強制避開樣本模擬
        fusion_alpha=0.0,
    )

    assert res["strategy"] in ("neighbor_lock", "pure_sample+global")
    assert res["predictions"]
    assert res["predictions"][0]["row"] == 0
    assert res["predictions"][0]["col"] == 1
    assert called["sim"] == 0


def test_neighbor_lock_fallback(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])

    # 模擬 neighbor 模組無作用，必定 fallback
    monkeypatch.setattr(
        analyzer,
        "neighbor_compatibility_score",
        lambda g, d: np.zeros_like(g, dtype=float),
    )
    monkeypatch.setattr(analyzer, "compute_neighbor_distribution", lambda *a, **k: {})

    called = {"sim": 0}

    def fake_sim(g, t, n_iter=0, **_):
        called["sim"] += 1
        return {
            (0, 1): {t: 0.6},
            (1, 1): {t: 0.4},
        }

    # ✅ 這裡關鍵！prior 熱圖無資料 → 模擬會被 fallback 掉
    monkeypatch.setattr(analyzer, "compute_position_probabilities", lambda *a, **k: {})

    monkeypatch.setattr(analyzer, "simulate_full_board", fake_sim)

    res = analyzer.predict_scratch_card(
        grid.tolist(),
        target_num=1,
        iterations=2,
        use_neighbor_lock=True,
        sample_gamma=1.0,  # 強制使用模擬路徑
        fusion_alpha=0.0,
    )

    assert res["strategy"] in ("neighbor_lock", "pure_sample+global")
    assert res["predictions"]
    assert res["predictions"][0]["col"] in (0, 1)
    assert called["sim"] == 1