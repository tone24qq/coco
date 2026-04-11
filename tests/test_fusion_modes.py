from __future__ import annotations

from src.inference_service import _compute_vote_scores, aggregate_candidate_scores


def _cands() -> list[dict]:
    return [
        {"cell": (0, 0), "module_scores": {"a": 0.9, "b": 0.6}, "module_details": {}, "score": 0.0},
        {"cell": (0, 1), "module_scores": {"a": 0.7, "b": 0.9}, "module_details": {}, "score": 0.0},
        {"cell": (1, 0), "module_scores": {"a": 0.2, "b": 0.1}, "module_details": {}, "score": 0.0},
    ]


def test_vote_scores_normalized() -> None:
    c = _cands()
    v = _compute_vote_scores(c, {"a": 0.5, "b": 0.5}, {})
    assert set(v.keys()) == {x["cell"] for x in c}
    assert all(0.0 <= s <= 1.0 for s in v.values())


def test_fusion_modes_produce_scores() -> None:
    for mode in ("weighted_only", "vote_only", "weighted_plus_vote", "weighted_plus_vote_with_gate"):
        c = _cands()
        diag = aggregate_candidate_scores(
            c,
            {"a": 0.5, "b": 0.5},
            {
                "type": "gate_then_weighted_sum",
                "fusion_mode": mode,
                "vote_alpha": 0.2,
                "gating_enabled": True,
            },
        )
        assert "fusion_mode" in diag
        assert all("score" in x for x in c)
