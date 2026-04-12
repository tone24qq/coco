from __future__ import annotations

from copy import deepcopy

from src.inference_service import _apply_spatial_cluster_penalty, _run_inference_detailed


def _board() -> list[list[int]]:
    return [
        [1, -1, -1, -1],
        [-1, 6, -1, -1],
        [-1, -1, 11, -1],
        [-1, -1, -1, 16],
    ]


def _agg_cfg(top_m: int, enabled: bool = True) -> dict:
    return {
        "type": "committee_weighted_sum",
        "weighting_mode": "yaml_normalized",
        "preserve_diagnostics": True,
        "spatial_postprocess": {
            "enabled": enabled,
            "method": "spatial_penalty",
            "distance_metric": "hybrid",
            "top_m": top_m,
            "penalty_d1": 0.10,
            "penalty_d2": 0.04,
            "score_gap_gate": 0.2,
            "mmr_lambda": 0.35,
            "mmr_distance_scale": 0.08,
            "mmr_d2_weight": 0.5,
            "protect_target_sensitive_threshold": 0.95,
            "protect_structure_threshold": 0.95,
            "protect_adjustment_threshold": 0.2,
            "protect_multiplier": 0.5,
            "max_penalty_per_candidate": 0.08,
        },
    }


def test_spatial_disabled_keeps_existing_result() -> None:
    board = _board()
    off = _run_inference_detailed(
        board,
        7,
        source="spatial_off",
        apply_reranker_stage=False,
        aggregator_config=_agg_cfg(5, False),
    )
    base = _run_inference_detailed(board, 7, source="spatial_base", apply_reranker_stage=False)
    assert [(c["row"], c["col"]) for c in off["candidate_cells"]] == [
        (c["row"], c["col"]) for c in base["candidate_cells"]
    ]
    assert [c["score"] for c in off["candidate_cells"]] == [c["score"] for c in base["candidate_cells"]]
    assert off["metadata"]["spatial_postprocess_enabled"] is False
    assert off["metadata"]["spatial_postprocess_applied"] is False


def test_spatial_top_m_accepts_3_5_8_10() -> None:
    board = _board()
    for top_m in (3, 5, 8, 10):
        out = _run_inference_detailed(
            board,
            7,
            source=f"spatial_top_m_{top_m}",
            apply_reranker_stage=False,
            aggregator_config=_agg_cfg(top_m, True),
        )
        assert out["metadata"]["spatial_postprocess_top_m"] == top_m


def test_spatial_postprocess_keeps_output_contract_consistent() -> None:
    out = _run_inference_detailed(
        _board(),
        7,
        source="spatial_contract",
        apply_reranker_stage=False,
        aggregator_config=_agg_cfg(5, True),
    )
    first = out["candidate_cells"][0]
    assert out["best_cell"]["row"] == first["row"]
    assert out["best_cell"]["col"] == first["col"]
    assert abs(out["best_ranking_score"] - float(first["score"])) < 1e-6
    assert out["metadata"]["final_top1_cell"] == (first["row"] - 1, first["col"] - 1)
    assert out["metadata"]["spatial_postprocess_method"] == "spatial_penalty"


def test_spatial_penalty_spreads_clustered_high_scores() -> None:
    ranked = [
        {
            "cell": (0, 0),
            "score": 0.90,
            "target_sensitive_score": 0.55,
            "module_scores": {},
            "assignment_delta": 0.0,
            "pairwise_delta": 0.0,
        },
        {
            "cell": (0, 1),
            "score": 0.89,
            "target_sensitive_score": 0.55,
            "module_scores": {},
            "assignment_delta": 0.0,
            "pairwise_delta": 0.0,
        },
        {
            "cell": (1, 1),
            "score": 0.88,
            "target_sensitive_score": 0.55,
            "module_scores": {},
            "assignment_delta": 0.0,
            "pairwise_delta": 0.0,
        },
        {
            "cell": (3, 3),
            "score": 0.87,
            "target_sensitive_score": 0.55,
            "module_scores": {},
            "assignment_delta": 0.0,
            "pairwise_delta": 0.0,
        },
    ]
    cfg = deepcopy(_agg_cfg(4, True))["spatial_postprocess"]
    cfg["score_gap_gate"] = 0.1
    out, diag = _apply_spatial_cluster_penalty(deepcopy(ranked), cfg)
    by_cell = {c["cell"]: c for c in out}
    assert diag["applied"] is True
    assert by_cell[(0, 1)]["spatial_cluster_penalty"] > 0.0
    assert by_cell[(1, 1)]["spatial_cluster_penalty"] > 0.0
    assert by_cell[(3, 3)].get("spatial_cluster_penalty", 0.0) == 0.0
    assert by_cell[(0, 1)]["score"] < 0.89


def test_evidence_aware_mmr_method_applies_penalty_and_reports_method() -> None:
    cfg = _agg_cfg(8, True)
    cfg["spatial_postprocess"]["method"] = "evidence_aware_mmr"
    out = _run_inference_detailed(
        _board(),
        7,
        source="spatial_mmr",
        apply_reranker_stage=False,
        aggregator_config=cfg,
    )
    assert out["metadata"]["spatial_postprocess_method"] == "evidence_aware_mmr"
    assert out["metadata"]["spatial_postprocess_applied"] is True
    assert any(float(c.get("spatial_cluster_penalty", 0.0)) > 0.0 for c in out["candidate_cells"][:8])
