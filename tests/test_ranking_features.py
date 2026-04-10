from __future__ import annotations

from src.ranking_features import FEATURE_SCHEMA_VERSION, build_candidate_feature_rows


def test_feature_schema_stable() -> None:
    candidates = [
        {
            "row": 1,
            "col": 1,
            "score": 0.9,
            "module_scores": {
                "logic_rule": 0.9,
                "pattern_model": 0.1,
                "prior_model": 0.5,
                "directional_consistency": 0.4,
                "line_consistency": 0.6,
                "global_assignment_prior": 0.3,
            },
            "module_details": {
                "directional_consistency": {"left_order_score": 0.5},
                "line_consistency": {"row_residual_score": 0.5},
            },
        },
        {
            "row": 1,
            "col": 2,
            "score": 0.7,
            "module_scores": {
                "logic_rule": 0.8,
                "pattern_model": 0.4,
                "prior_model": 0.2,
                "directional_consistency": 0.5,
                "line_consistency": 0.7,
                "global_assignment_prior": 0.4,
            },
            "module_details": {
                "directional_consistency": {"left_order_score": 0.7},
                "line_consistency": {"row_residual_score": 0.7},
            },
        },
    ]
    rows = build_candidate_feature_rows(
        "c1",
        (2, 2),
        candidates,
        true_cell_1_based=(1, 2),
        board=[[1, -1], [-1, 4]],
        target_number=2,
    )
    assert rows[0]["case_id"] == "c1"
    assert rows[0]["module_consensus_top3"] >= 0
    assert rows[0]["baseline_rank"] == 1
    assert "directional_score" in rows[0]
    assert "line_score" in rows[0]
    assert "global_assignment_score" in rows[0]
    assert rows[1]["label"] == 1
    assert FEATURE_SCHEMA_VERSION.startswith("ranking_features_")


def test_relative_rank_dense_with_duplicate_scores() -> None:
    candidates = [
        {"row": 1, "col": 1, "score": 0.9, "module_scores": {}},
        {"row": 1, "col": 2, "score": 0.9, "module_scores": {}},
        {"row": 1, "col": 3, "score": 0.7, "module_scores": {}},
    ]
    rows = build_candidate_feature_rows("dup", (1, 3), candidates)
    assert rows[0]["relative_rank_within_row"] == 1
    assert rows[1]["relative_rank_within_row"] == 1
    assert rows[2]["relative_rank_within_row"] == 2
