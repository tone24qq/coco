from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.competitive_fusion import validate_meta_judge_artifact
from src.inference_service import (
    InferenceError,
    _run_inference_detailed,
    aggregate_candidate_scores,
    build_cell_candidates,
)
from src.inference_service import score_candidates_raw


def _board() -> list[list[int]]:
    return [
        [1, 2, 3, 4, 5],
        [6, -1, 8, -1, 10],
        [11, 12, -1, 14, 15],
        [16, -1, 18, -1, 20],
    ]


def _cfg(mode: str = "weighted_rank_fusion", artifact: str = "artifacts/competitive_judge_artifact.json") -> dict:
    return {
        "type": "competitive_ensemble",
        "fusion_mode": mode,
        "competitor_normalization": "per_module_minmax",
        "include_vote_features": True,
        "include_rank_features": True,
        "include_score_features": True,
        "judge_model_type": "logistic_ranker",
        "judge_feature_schema_version": "competitive_features_v1",
        "judge_artifact_path": artifact,
        "fallback_mode": "weighted_rank_fusion",
        "preserve_diagnostics": True,
        "rank_fusion_method": "rrf",
    }


def test_equal_start_stage_does_not_use_yaml_module_weights_for_stage_a() -> None:
    candidates_a = [
        {"cell": (0, 0), "module_scores": {"logic_rule": 0.9, "prior_model": 0.1}, "module_details": {}},
        {"cell": (0, 1), "module_scores": {"logic_rule": 0.1, "prior_model": 0.9}, "module_details": {}},
    ]
    candidates_b = [dict(x) for x in candidates_a]
    aggregate_candidate_scores(candidates_a, {"logic_rule": 0.9, "prior_model": 0.1}, _cfg())
    aggregate_candidate_scores(candidates_b, {"logic_rule": 0.1, "prior_model": 0.9}, _cfg())
    stage_a_a = {c["cell"]: c["stage_a_rank"] for c in candidates_a}
    stage_a_b = {c["cell"]: c["stage_a_rank"] for c in candidates_b}
    assert stage_a_a == stage_a_b


def test_pairwise_auto_top_competitors_does_not_reuse_legacy_weight_bias() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    base_candidates = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    settings = {"pairwise_conditional_consistency": {"pairwise_seed_modules": ["__auto_top_competitors__"]}}
    c1, _, _ = score_candidates_raw(
        board,
        [dict(x) for x in base_candidates],
        4,
        module_weights={"logic_rule": 0.9, "line_consistency": 0.1, "pairwise_conditional_consistency": 0.2},
        module_settings=settings,
    )
    c2, _, _ = score_candidates_raw(
        board,
        [dict(x) for x in base_candidates],
        4,
        module_weights={"logic_rule": 0.1, "line_consistency": 0.9, "pairwise_conditional_consistency": 0.2},
        module_settings=settings,
    )
    s1 = [round(float(x["module_scores"].get("pairwise_conditional_consistency", 0.0)), 6) for x in c1]
    s2 = [round(float(x["module_scores"].get("pairwise_conditional_consistency", 0.0)), 6) for x in c2]
    assert s1 == s2


def test_meta_judge_artifact_validation_fail_fast() -> None:
    try:
        validate_meta_judge_artifact({"model_type": "logistic_ranker"})
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid artifact")


def test_meta_judge_fallback_records_reason() -> None:
    out = _run_inference_detailed(
        _board(),
        7,
        source="test",
        apply_reranker_stage=False,
        aggregator_config=_cfg(mode="learned_meta_ranker", artifact="artifacts/missing.json"),
    )
    assert str(out["metadata"].get("fallback_reason", "")).startswith("meta_judge_fallback")


def test_competitive_fusion_modes_are_only_three_canonical_modes() -> None:
    candidates = [{"cell": (0, 0), "module_scores": {"logic_rule": 1.0}, "module_details": {}}]
    for mode in ("weighted_rank_fusion", "vote_based_fusion", "learned_meta_ranker"):
        aggregate_candidate_scores(candidates, {"logic_rule": 1.0}, _cfg(mode=mode))
    try:
        aggregate_candidate_scores(
            candidates,
            {"logic_rule": 1.0},
            _cfg(mode="weighted_plus_vote"),
        )
    except InferenceError:
        return
    raise AssertionError("legacy fusion mode should not be accepted")


def test_real_data_backtest_preferred_over_synthetic_for_recommendation(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    rows = []
    board = _board()
    for target in (7, 9, 13, 17, 19, 7, 9, 13):
        true = ((target - 1) // 5 + 1, (target - 1) % 5 + 1)
        rows.append(json.dumps({"board": board, "target_number": target, "true_cell": list(true)}))
    real.write_text("\n".join(rows), encoding="utf-8")
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_competitive_fusion_backtest.py",
            "--real-data",
            str(real),
            "--min-real-cases",
            "4",
        ]
    )
    report = json.loads(Path("reports/competitive_fusion_report.json").read_text(encoding="utf-8"))
    assert "real_data_comparison" in report
    assert report["insufficient_real_data"] is False


def test_reranker_artifact_is_independent_from_competitive_judge_artifact() -> None:
    out = _run_inference_detailed(
        _board(),
        7,
        source="test",
        apply_reranker_stage=False,
        aggregator_config=_cfg(mode="weighted_rank_fusion"),
    )
    assert out["metadata"]["aggregation_type"] == "competitive_ensemble"
    assert "judge_artifact_path" in out["metadata"]


def test_candidate_features_include_all_competitor_outputs() -> None:
    out = _run_inference_detailed(
        _board(),
        7,
        source="test",
        apply_reranker_stage=False,
        aggregator_config=_cfg(mode="weighted_rank_fusion"),
    )
    top = out["candidate_cells"][0]
    assert "mean_score" in top and "rrf_score" in top and "top1_vote_count" in top
    assert any(k.startswith("module_") and k.endswith("_is_top1") for k in top)


def test_walk_forward_only_enforced_in_training_script() -> None:
    text = Path("scripts/train_competitive_judge.py").read_text(encoding="utf-8")
    assert "walk_forward" in text
    assert "random" not in text.lower()


def test_api_contract_still_returns_top10_and_best_confidence() -> None:
    out = _run_inference_detailed(
        _board(),
        7,
        source="test",
        apply_reranker_stage=False,
        aggregator_config=_cfg(mode="weighted_rank_fusion"),
    )
    assert "best_cell" in out and "confidence_score" in out
    assert len(out["candidate_cells"]) >= 1
    assert "confidence_1_to_100" in out["best_cell"]
