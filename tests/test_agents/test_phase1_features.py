from src.build_features import build_feature_rows


def test_feature_contract_has_80_candidates(synthetic_records) -> None:
    rows = build_feature_rows(synthetic_records, min_history=100, min_dynamic_n=20, max_dynamic_n=999, top_k=8)
    first_issue = rows[0]["issue"]
    assert sum(1 for r in rows if r["issue"] == first_issue) == 80
    first = rows[0]
    assert first["cand_avg_gap"] > 0
    assert "issue_transition_regime" in first
    assert "retrieval_next_draw_posterior" in first
