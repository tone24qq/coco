from src.build_features import build_feature_rows


def test_feature_contract_has_80_candidates(synthetic_records) -> None:
    rows = build_feature_rows(synthetic_records, min_history=100, retrieval_window=40, top_k=8)
    assert len(rows) == (len(synthetic_records) - 100) * 80
    first = rows[0]
    assert first["cand_avg_gap"] > 0
    assert "issue_transition_regime" in first
    assert first["label"] in {0, 1}
