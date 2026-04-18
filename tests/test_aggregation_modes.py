from __future__ import annotations

from src.inference_service import aggregate_candidate_scores


def _candidate(a: float, b: float) -> dict:
    return {
        "cell": (0, 0),
        "score": 0.0,
        "module_scores": {"m1": a, "m2": b},
        "module_informative": {"m1": 1.0, "m2": 1.0},
        "module_details": {},
    }


def test_hard_gate_abstains_low_scores() -> None:
    cands = [_candidate(0.4, 0.9)]
    diag = aggregate_candidate_scores(
        cands,
        {"m1": 0.5, "m2": 0.5},
        {
            "type": "committee_weighted_sum",
            "weighting_mode": "yaml_normalized",
            "contribution_mode": "weighted_sum",
            "confidence_gate_threshold": 0.5,
            "abstain_below_threshold": True,
            "low_confidence_weight_multiplier": 1.0,
        },
    )
    assert cands[0]["abstain_module_count"] == 1
    assert cands[0]["active_module_count"] >= 1
    assert diag["abstain_rate"] > 0


def test_centered_weighted_sum_supports_negative_votes() -> None:
    cands = [_candidate(0.2, 0.8)]
    aggregate_candidate_scores(
        cands,
        {"m1": 0.5, "m2": 0.5},
        {
            "type": "committee_weighted_sum",
            "weighting_mode": "yaml_normalized",
            "contribution_mode": "centered_weighted_sum",
            "use_centered_score": True,
            "confidence_gate_threshold": 0.5,
            "abstain_below_threshold": False,
            "low_confidence_weight_multiplier": 1.0,
        },
    )
    # 0.2 gives negative vote, 0.8 gives positive vote, net score should stay around neutral 0.5
    assert 0.49 <= float(cands[0]["stage1_base_score"]) <= 0.51
