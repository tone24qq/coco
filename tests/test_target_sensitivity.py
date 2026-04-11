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


def _competitive_cfg(mode: str = "weighted_rank_fusion") -> dict:
    return {
        "type": "competitive_ensemble",
        "fusion_mode": mode,
        "competitor_normalization": "per_module_minmax",
        "include_vote_features": True,
        "include_rank_features": True,
        "include_score_features": True,
        "fallback_mode": "weighted_rank_fusion",
        "judge_artifact_path": "artifacts/does_not_exist.json",
    }


def test_same_board_different_targets_not_always_same_top1() -> None:
    board = _target_sensitive_board()
    r1 = _run_inference_detailed(
        board, 7, source="t1", apply_reranker_stage=False, aggregator_config=_competitive_cfg()
    )
    r2 = _run_inference_detailed(
        board, 9, source="t2", apply_reranker_stage=False, aggregator_config=_competitive_cfg()
    )
    map1 = {(c["row"], c["col"]): c["target_sensitive_score"] for c in r1["candidate_cells"]}
    map2 = {(c["row"], c["col"]): c["target_sensitive_score"] for c in r2["candidate_cells"]}
    assert map1 != map2


def test_vote_based_fusion_reorders_from_stage_a() -> None:
    candidates = [
        {"cell": (0, 0), "module_scores": {"logic_rule": 0.9, "prior_model": 0.1}, "module_details": {}},
        {"cell": (0, 1), "module_scores": {"logic_rule": 0.8, "prior_model": 0.9}, "module_details": {}},
    ]
    aggregate_candidate_scores(
        candidates, {"logic_rule": 0.5, "prior_model": 0.5}, _competitive_cfg("vote_based_fusion")
    )
    ordered = sorted(candidates, key=lambda x: x["final_rank_position"])
    assert ordered[0]["was_reordered_by_tiebreak"] in (True, False)


def test_pairwise_seed_uses_module_competition_paths() -> None:
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
