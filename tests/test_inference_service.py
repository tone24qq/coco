from __future__ import annotations

from src.inference_service import (
    _normalize_scores,
    aggregate_candidate_scores,
    build_cell_candidates,
    map_best_confidence_1_100,
    rank_candidates,
    run_inference,
    score_candidates,
)


def test_module_weights_take_effect() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    candidates = build_cell_candidates([(0, 1), (1, 0), (1, 2)])

    scored_logic, _, _ = score_candidates(
        board,
        [
            {"cell": c["cell"], "score": 0.0, "module_scores": {}}
            for c in candidates
        ],
        target_number=4,
        module_weights={"logic_rule": 1.0},
    )
    scored_prior, _, _ = score_candidates(
        board,
        [
            {"cell": c["cell"], "score": 0.0, "module_scores": {}}
            for c in candidates
        ],
        target_number=4,
        module_weights={"prior_model": 1.0},
    )

    top_logic = rank_candidates(scored_logic)[0]["cell"]
    top_prior = rank_candidates(scored_prior)[0]["cell"]
    assert top_logic != top_prior


def test_module_settings_applied_even_when_module_weights_passed() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    candidates = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    scored, _, _ = score_candidates(
        board,
        candidates,
        target_number=4,
        module_weights={"global_assignment_prior": 1.0},
        module_settings={"global_assignment_prior": {"assignment_mode": "greedy"}},
    )
    assert "global_assignment_prior" in scored[0]["module_details"]
    assert scored[0]["module_details"]["global_assignment_prior"]["global_assignment_mode"] == 0.0


def test_no_global_module_state_leakage_between_calls() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    c1 = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    out1, _, _ = score_candidates(
        board,
        c1,
        target_number=4,
        module_weights={"global_assignment_prior": 1.0},
        module_settings={"global_assignment_prior": {"assignment_mode": "greedy"}},
    )
    c2 = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    out2, _, _ = score_candidates(
        board,
        c2,
        target_number=4,
        module_weights={"global_assignment_prior": 1.0},
        module_settings={"global_assignment_prior": {"assignment_mode": "exact"}},
    )
    assert out1[0]["module_details"]["global_assignment_prior"]["global_assignment_mode"] == 0.0
    assert out2[0]["module_details"]["global_assignment_prior"]["global_assignment_mode"] == 1.0


def test_constant_module_scores_are_neutral_not_all_ones() -> None:
    out = _normalize_scores({(0, 0): 2.0, (0, 1): 2.0}, mode="minmax")
    assert out[(0, 0)] == 0.5
    assert out[(0, 1)] == 0.5


def test_normalization_disabled_keeps_weak_signals_close() -> None:
    raw = {(0, 0): 0.51, (0, 1): 0.5, (1, 1): 0.49}
    out = _normalize_scores(raw, mode="disabled")
    assert out == raw


def test_normalization_light_mode_runs_and_preserves_ordering() -> None:
    raw = {(0, 0): 0.51, (0, 1): 0.5, (1, 1): 0.49}
    out = _normalize_scores(raw, mode="light")
    assert out[(0, 0)] > out[(0, 1)] > out[(1, 1)]


def test_hard_gate_can_eliminate_high_support_candidate() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "score": 0.0,
            "module_scores": {"logic_rule": 0.9},
            "module_details": {
                "directional_consistency": {"row_violation_count": 2.0, "col_violation_count": 1.0},
                "line_consistency": {
                    "diag_violation_count": 0.0,
                    "monotonic_break_flag": 1.0,
                    "percentile_outlier_flag": 1.0,
                    "gap_outlier_flag": 0.0,
                },
            },
        },
        {
            "cell": (0, 1),
            "score": 0.0,
            "module_scores": {"logic_rule": 0.6},
            "module_details": {
                "directional_consistency": {"row_violation_count": 0.0, "col_violation_count": 0.0},
                "line_consistency": {
                    "diag_violation_count": 0.0,
                    "monotonic_break_flag": 0.0,
                    "percentile_outlier_flag": 0.0,
                    "gap_outlier_flag": 0.0,
                },
            },
        },
    ]
    aggregate_candidate_scores(
        candidates,
        {"logic_rule": 1.0},
        {"type": "gate_then_weighted_sum", "gating_enabled": True, "hard_violation_threshold": 2.0},
    )
    ranked = rank_candidates(candidates)
    assert ranked[0]["cell"] == (0, 1)


def test_contradiction_penalty_changes_ranking() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "score": 0.0,
            "module_scores": {"logic_rule": 0.7, "line_consistency": 0.7},
            "module_details": {"logic_rule": {"local_contradiction_penalty": 0.8}, "line_consistency": {}},
        },
        {
            "cell": (0, 1),
            "score": 0.0,
            "module_scores": {"logic_rule": 0.65, "line_consistency": 0.65},
            "module_details": {"logic_rule": {"local_contradiction_penalty": 0.0}, "line_consistency": {}},
        },
    ]
    aggregate_candidate_scores(
        candidates,
        {"logic_rule": 0.5, "line_consistency": 0.5},
        {"type": "gate_then_weighted_sum", "gating_enabled": True, "score_spread_enabled": True},
    )
    assert rank_candidates(candidates)[0]["cell"] == (0, 1)


def test_score_spread_expands_but_preserves_ranking_order() -> None:
    candidates = [
        {"cell": (0, 0), "score": 0.0, "module_scores": {"logic_rule": 0.501}, "module_details": {}},
        {"cell": (0, 1), "score": 0.0, "module_scores": {"logic_rule": 0.5}, "module_details": {}},
        {"cell": (1, 1), "score": 0.0, "module_scores": {"logic_rule": 0.499}, "module_details": {}},
    ]
    diagnostics = aggregate_candidate_scores(
        candidates,
        {"logic_rule": 1.0},
        {"type": "gate_then_weighted_sum", "gating_enabled": False, "score_spread_enabled": True},
    )
    ranked = rank_candidates(candidates)
    assert [c["cell"] for c in ranked] == [(0, 0), (0, 1), (1, 1)]
    assert diagnostics["final_score_std"] > diagnostics["raw_score_std"]


def test_candidate_confidence_not_all_identical() -> None:
    result = run_inference([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=False)
    confs = {c["confidence_1_to_100"] for c in result["candidate_cells"]}
    assert len(confs) > 1


def test_collapsed_score_flag_true_on_flat_scores() -> None:
    candidates = [
        {"cell": (0, 0), "score": 0.0, "module_scores": {"logic_rule": 0.5}, "module_details": {}},
        {"cell": (0, 1), "score": 0.0, "module_scores": {"logic_rule": 0.5}, "module_details": {}},
    ]
    diagnostics = aggregate_candidate_scores(
        candidates,
        {"logic_rule": 1.0},
        {"type": "gate_then_weighted_sum", "gating_enabled": False, "score_spread_enabled": False},
    )
    assert diagnostics["collapsed_score_flag"] is True


def test_best_confidence_rises_with_larger_margin() -> None:
    low = map_best_confidence_1_100(
        margin_to_top2=0.01,
        top1_top5_mean_gap=0.01,
        effective_candidate_count=5,
        gated_candidate_count=5,
        score_entropy_like=0.95,
        collapsed_score_flag=True,
    )
    high = map_best_confidence_1_100(
        margin_to_top2=0.2,
        top1_top5_mean_gap=0.15,
        effective_candidate_count=5,
        gated_candidate_count=2,
        score_entropy_like=0.4,
        collapsed_score_flag=False,
    )
    assert high > low
