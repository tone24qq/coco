from __future__ import annotations

from pathlib import Path

from src.inference_service import _run_inference_detailed, aggregate_candidate_scores
from src.inference_service import build_cell_candidates, score_candidates


def _board() -> list[list[int]]:
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


def test_competitive_fusion_runs_with_all_enabled_modules() -> None:
    board = _board()
    out = _run_inference_detailed(
        board, 7, source="test", apply_reranker_stage=False, aggregator_config=_competitive_cfg()
    )
    assert out["status"] == "ok"
    assert len(out["candidate_cells"]) > 0
    assert out["metadata"]["ranking_contract_version"] == "competitive_ensemble_v1"


def test_no_primary_tiebreak_privilege_remains() -> None:
    board = _board()
    out = _run_inference_detailed(
        board, 7, source="test", apply_reranker_stage=False, aggregator_config=_competitive_cfg()
    )
    meta = out["metadata"]
    assert "target_primary_modules" not in meta
    assert "tie_break_modules" not in meta


def test_meta_judge_fallback_when_artifact_missing() -> None:
    board = _board()
    out = _run_inference_detailed(
        board,
        7,
        source="test",
        apply_reranker_stage=False,
        aggregator_config=_competitive_cfg(mode="learned_meta_ranker"),
    )
    assert str(out["metadata"].get("fallback_reason", "")).startswith("meta_judge_fallback")


def test_rank_fusion_output_is_deterministic() -> None:
    board = _board()
    cfg = _competitive_cfg(mode="weighted_rank_fusion")
    out1 = _run_inference_detailed(board, 7, source="test", apply_reranker_stage=False, aggregator_config=cfg)
    out2 = _run_inference_detailed(board, 7, source="test", apply_reranker_stage=False, aggregator_config=cfg)
    top10_1 = [(c["row"], c["col"]) for c in out1["candidate_cells"][:10]]
    top10_2 = [(c["row"], c["col"]) for c in out2["candidate_cells"][:10]]
    assert top10_1 == top10_2


def test_candidate_feature_schema_contains_all_module_features() -> None:
    board = _board()
    out = _run_inference_detailed(
        board, 7, source="test", apply_reranker_stage=False, aggregator_config=_competitive_cfg()
    )
    top = out["candidate_cells"][0]
    assert any(k.startswith("module_") and k.endswith("_score") for k in top)
    assert any(k.startswith("module_") and k.endswith("_rank") for k in top)
    assert "top1_vote_count" in top
    assert "rrf_score" in top


def test_walk_forward_split_only_no_random_split() -> None:
    path = Path("scripts/run_competitive_fusion_backtest.py")
    text = path.read_text(encoding="utf-8")
    assert "walk_forward" in text
    assert "random" not in text.lower()


def test_api_contract_unchanged_for_predict_output() -> None:
    board = _board()
    out = _run_inference_detailed(
        board, 7, source="test", apply_reranker_stage=False, aggregator_config=_competitive_cfg()
    )
    assert "best_cell" in out
    assert "candidate_cells" in out
    assert "confidence_score" in out
    assert "ranking_score" in out["candidate_cells"][0]


def test_competitive_fusion_preserves_module_diagnostics() -> None:
    board = _board()
    candidates = build_cell_candidates([(1, 1), (1, 3), (2, 2)])
    scored, weights, _ = score_candidates(board, candidates, 7)
    aggregate_candidate_scores(scored, weights, _competitive_cfg())
    first = scored[0]
    assert "module_scores" in first
    assert "module_details" in first
    assert "mean_score" in first
