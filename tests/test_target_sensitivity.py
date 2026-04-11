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


def test_same_board_different_targets_not_always_same_top1() -> None:
    board = _target_sensitive_board()
    r1 = _run_inference_detailed(board, 7, source="t1", apply_reranker_stage=False)
    r2 = _run_inference_detailed(board, 9, source="t2", apply_reranker_stage=False)
    top1_a = (r1["candidate_cells"][0]["row"], r1["candidate_cells"][0]["col"])
    top1_b = (r2["candidate_cells"][0]["row"], r2["candidate_cells"][0]["col"])
    assert top1_a != top1_b


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
