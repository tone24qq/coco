from __future__ import annotations

from src.inference_service import aggregate_candidate_scores


def _cands() -> list[dict]:
    return [
        {
            "cell": (0, 0),
            "module_scores": {"logic_rule": 0.9, "directional_consistency": 0.6},
            "module_details": {},
            "score": 0.0,
        },
        {
            "cell": (0, 1),
            "module_scores": {"logic_rule": 0.7, "directional_consistency": 0.9},
            "module_details": {},
            "score": 0.0,
        },
        {
            "cell": (1, 0),
            "module_scores": {"logic_rule": 0.2, "directional_consistency": 0.1},
            "module_details": {},
            "score": 0.0,
        },
    ]


def _cfg(mode: str) -> dict:
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


def test_fusion_modes_produce_scores() -> None:
    for mode in ("weighted_rank_fusion", "vote_based_fusion", "learned_meta_ranker"):
        c = _cands()
        diag = aggregate_candidate_scores(c, {"logic_rule": 0.5, "directional_consistency": 0.5}, _cfg(mode))
        assert "fusion_mode" in diag
        assert all("score" in x for x in c)


def test_rank_features_present() -> None:
    c = _cands()
    aggregate_candidate_scores(c, {"logic_rule": 0.5, "directional_consistency": 0.5}, _cfg("weighted_rank_fusion"))
    assert "module_logic_rule_rank" in c[0]
    assert "module_directional_consistency_is_top3" in c[0]
