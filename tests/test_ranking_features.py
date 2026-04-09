from __future__ import annotations

from src.ranking_features import FEATURE_SCHEMA_VERSION, build_candidate_feature_rows


def test_feature_schema_stable() -> None:
    candidates = [
        {
            "row": 1,
            "col": 1,
            "score": 0.9,
            "module_scores": {"logic_rule": 0.9, "pattern_model": 0.1, "prior_model": 0.5},
        },
        {
            "row": 1,
            "col": 2,
            "score": 0.7,
            "module_scores": {"logic_rule": 0.8, "pattern_model": 0.4, "prior_model": 0.2},
        },
    ]
    rows = build_candidate_feature_rows("c1", (2, 2), candidates, true_cell_1_based=(1, 2))
    assert rows[0]["case_id"] == "c1"
    assert rows[0]["module_consensus_top3"] >= 0
    assert rows[0]["baseline_rank"] == 1
    assert rows[1]["label"] == 1
    assert FEATURE_SCHEMA_VERSION.startswith("ranking_features_")
