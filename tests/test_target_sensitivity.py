from __future__ import annotations

from src.inference_service import InferenceError, _run_inference_detailed, aggregate_candidate_scores
from src.inference_service import build_cell_candidates, score_candidates
from src.scoring_modules import PairwiseConditionalConsistencyModule


def _target_sensitive_board() -> list[list[int]]:
    return [
        [1, 2, 3, 4, 5],
        [6, -1, 8, -1, 10],
        [11, 12, -1, 14, 15],
        [16, -1, 18, -1, 20],
    ]


def _regression_fixture_board() -> list[list[int]]:
    rows, cols = 8, 10
    board = []
    n = 1
    masked = {13, 16, 22, 25, 31, 34, 40, 43, 49, 52, 58, 61, 67, 70, 76, 79}
    for _r in range(rows):
        row = []
        for _c in range(cols):
            row.append(-1 if n in masked else n)
            n += 1
        board.append(row)
    return board


def test_same_board_different_targets_not_always_same_top1() -> None:
    board = _target_sensitive_board()
    r1 = _run_inference_detailed(board, 7, source="t1", apply_reranker_stage=False)
    r2 = _run_inference_detailed(board, 9, source="t2", apply_reranker_stage=False)
    map1 = {(c["row"], c["col"]): c["target_primary_score"] for c in r1["candidate_cells"]}
    map2 = {(c["row"], c["col"]): c["target_primary_score"] for c in r2["candidate_cells"]}
    assert map1 != map2
    top1_a = (r1["candidate_cells"][0]["row"], r1["candidate_cells"][0]["col"])
    top1_b = (r2["candidate_cells"][0]["row"], r2["candidate_cells"][0]["col"])
    assert top1_a != top1_b


def test_regression_fixture_target13_16_top5_not_identical() -> None:
    board = _regression_fixture_board()
    r13 = _run_inference_detailed(board, 13, source="r13", apply_reranker_stage=False)
    r16 = _run_inference_detailed(board, 16, source="r16", apply_reranker_stage=False)
    top5_13 = [(c["row"], c["col"]) for c in r13["candidate_cells"][:5]]
    top5_16 = [(c["row"], c["col"]) for c in r16["candidate_cells"][:5]]
    assert top5_13 != top5_16


def test_stage_a_margin_locks_top1() -> None:
    candidates = [
        {"cell": (0, 0), "module_scores": {"logic_rule": 0.95, "prior_model": 0.1}, "module_details": {}},
        {"cell": (0, 1), "module_scores": {"logic_rule": 0.70, "prior_model": 0.9}, "module_details": {}},
    ]
    aggregate_candidate_scores(
        candidates,
        {"logic_rule": 0.9, "prior_model": 0.1},
        {
            "fusion_mode": "weighted_plus_vote_with_gate",
            "target_primary_modules": ["logic_rule"],
            "target_agnostic_modules": ["prior_model"],
            "tie_break_modules": ["prior_model"],
            "vote_include_modules": ["logic_rule"],
            "epsilon_primary": 0.02,
            "max_target_agnostic_weight_share": 0.2,
        },
    )
    ordered = sorted(candidates, key=lambda x: x["final_rank_position"])
    assert ordered[0]["cell"] == (0, 0)
    assert ordered[0]["primary_locked_top1"] is True


def test_tiebreak_can_reorder_when_within_epsilon() -> None:
    candidates = [
        {"cell": (0, 0), "module_scores": {"logic_rule": 0.80, "prior_model": 0.1}, "module_details": {}},
        {"cell": (0, 1), "module_scores": {"logic_rule": 0.79, "prior_model": 0.9}, "module_details": {}},
    ]
    diag = aggregate_candidate_scores(
        candidates,
        {"logic_rule": 0.9, "prior_model": 0.1},
        {
            "fusion_mode": "weighted_plus_vote_with_gate",
            "target_primary_modules": ["logic_rule"],
            "target_agnostic_modules": ["prior_model"],
            "tie_break_modules": ["prior_model"],
            "vote_include_modules": ["logic_rule"],
            "epsilon_primary": 0.05,
            "max_target_agnostic_weight_share": 0.3,
        },
    )
    ordered = sorted(candidates, key=lambda x: x["final_rank_position"])
    assert diag["top1_changed_by_tiebreak"] is True
    assert ordered[0]["was_reordered_by_tiebreak"] is True


def test_vote_uses_only_configured_modules() -> None:
    candidates = [
        {"cell": (0, 0), "module_scores": {"logic_rule": 0.9, "prior_model": 0.1}, "module_details": {}},
        {"cell": (0, 1), "module_scores": {"logic_rule": 0.1, "prior_model": 0.9}, "module_details": {}},
    ]
    weights = {"logic_rule": 0.5, "prior_model": 0.5}
    out = aggregate_candidate_scores(
        candidates,
        weights,
        {
            "type": "gate_then_weighted_sum",
            "fusion_mode": "vote_only",
            "vote_include_modules": ["logic_rule"],
            "target_primary_modules": ["logic_rule"],
            "target_sensitive_modules": ["logic_rule"],
            "target_agnostic_modules": ["prior_model"],
            "max_target_agnostic_weight_share": 0.2,
        },
    )
    assert out["fusion_mode"] == "vote_only"
    assert candidates[0]["vote_bonus"] > candidates[1]["vote_bonus"]


def test_vote_include_modules_fail_fast_on_unknown_module() -> None:
    candidates = [{"cell": (0, 0), "module_scores": {"logic_rule": 1.0}, "module_details": {}}]
    try:
        aggregate_candidate_scores(
            candidates,
            {"logic_rule": 1.0},
            {
                "type": "gate_then_weighted_sum",
                "vote_include_modules": ["unknown"],
                "target_primary_modules": ["logic_rule"],
            },
        )
    except InferenceError:
        return
    raise AssertionError("Expected InferenceError for unknown vote module")


def test_pairwise_seed_comes_from_target_sensitive_backbone() -> None:
    module = PairwiseConditionalConsistencyModule(runtime_mode="fast", candidate_top_n=1)
    module.set_seed_ranked_candidates([(0, 1)])
    board = [[1, -1, 3], [-1, 5, -1]]
    unopened = [(0, 1), (1, 0), (1, 2)]
    out = module.score(board, unopened, target_number=4)
    reduced = [c for c in unopened if out.details[c].get("runtime_reduced_path", 0.0) > 0.0]
    assert len(reduced) >= 1


def test_pairwise_seed_modules_empty_fail_fast() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    candidates = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    try:
        score_candidates(
            board,
            candidates,
            4,
            module_weights={
                "logic_rule": 0.5,
                "pairwise_conditional_consistency": 0.5,
            },
            module_settings={"pairwise_conditional_consistency": {"pairwise_seed_modules": []}},
        )
    except InferenceError:
        return
    raise AssertionError("Expected InferenceError for empty pairwise_seed_modules")
