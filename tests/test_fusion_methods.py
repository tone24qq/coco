from fusion import borda_rank, fuse_scores_dynamic


def test_fuse_scores_dynamic_basic():
    val = fuse_scores_dynamic({"conn": 0.9, "focus": None, "tail": 0.1})
    assert 0.0 <= val <= 1.0


def test_fuse_scores_dynamic_conn_boost(monkeypatch):
    """當 conn > 0.9 其他模組低時，boost 應生效。"""
    monkeypatch.setenv("FUSION_CONN_BOOST", "0.3")
    val = fuse_scores_dynamic({"conn": 0.95, "focus": 0.05})
    assert 0.95 < val <= 1.0  # 應明顯高於 0.95
    monkeypatch.delenv("FUSION_CONN_BOOST", raising=False)


def test_borda_rank_output():
    module_maps = {
        "A": {(0, 0): 0.9, (0, 1): 0.1},
        "B": {(0, 1): 0.8, (0, 0): 0.2},
    }
    ranks = borda_rank(module_maps, top_n=2)
    assert len(ranks) == 2
    assert ranks[0][1] >= ranks[1][1]
