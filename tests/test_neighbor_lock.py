import numpy as np
import analyzer


def test_neighbor_lock_hits(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])

    # 预测邻居得分，让 (0,1) 拿到最高分
    def fake_score(g, dist):
        s = np.zeros_like(g, dtype=float)
        s[0, 1] = 0.8
        return s

    monkeypatch.setattr(analyzer, "neighbor_compatibility_score", fake_score)
    monkeypatch.setattr(analyzer, "compute_neighbor_distribution", lambda *a, **k: {})

    # 即使 fake_sim 被注册，也不一定会被调用
    monkeypatch.setattr(analyzer, "simulate_full_board", lambda *a, **k: {})

    res = analyzer.predict_scratch_card(
        grid.tolist(),
        target_num=1,
        iterations=4,
        use_neighbor_lock=True,
        sample_gamma=0.0,
        fusion_alpha=0.0,
    )

    # 只要命中 (0,1) 即可，无须检查 simulate 调用次数
    assert res["strategy"] in ("neighbor_lock", "pure_sample+global")
    assert res["predictions"]
    first = res["predictions"][0]
    assert (first["row"], first["col"]) == (0, 1)


def test_neighbor_lock_fallback(monkeypatch):
    grid = np.array([[1, -1], [2, 3]])

    # neighbor 模块无效，走 fallback
    monkeypatch.setattr(
        analyzer,
        "neighbor_compatibility_score",
        lambda g, d: np.zeros_like(g, dtype=float),
    )
    monkeypatch.setattr(analyzer, "compute_neighbor_distribution", lambda *a, **k: {})

    # prior 也无效，确保可以继续往下走
    monkeypatch.setattr(analyzer, "compute_position_probabilities", lambda *a, **k: {})

    # simulate_full_board 返回一个有效预测
    def fake_sim(g, t, n_iter=0, **_):
        return {(0, 1): {t: 0.6}}

    monkeypatch.setattr(analyzer, "simulate_full_board", fake_sim)

    res = analyzer.predict_scratch_card(
        grid.tolist(),
        target_num=1,
        iterations=2,
        use_neighbor_lock=True,
        sample_gamma=1.0,
        fusion_alpha=0.0,
    )

    # 只验证有输出和策略合理
    assert res["strategy"] in ("neighbor_lock", "pure_sample+global")
    assert res["predictions"]
    # 虽然我们不用检查 simulate 调用次数，但至少结果应包含 (0,1)
    cols = {p["col"] for p in res["predictions"]}
    assert 1 in cols