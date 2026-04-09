from __future__ import annotations

import json
from pathlib import Path

from src.inference_service import run_inference


def test_reranker_fallback_when_artifact_missing(tmp_path: Path) -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    target = artifacts / "reranker_weights.json"
    backup = None
    if target.exists():
        backup = target.read_text()
        target.unlink()

    result = run_inference([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=True)
    assert result["metadata"]["ranking_stage"] == "baseline_only"
    assert result["metadata"]["reranker_fallback_reason"] is not None

    if backup is not None:
        target.write_text(backup)


def test_reranker_keeps_candidate_set() -> None:
    artifact_path = Path("artifacts/reranker_weights.json")
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "enabled": True,
                "version": "test",
                "feature_schema_version": "ranking_features_v1",
                "feature_columns": ["baseline_score"],
                "weights": {"baseline_score": 1.0},
            }
        )
    )
    baseline = run_inference([[1, -1, 3], [-1, 5, -1]], 4, source="b", apply_reranker_stage=False)
    reranked = run_inference([[1, -1, 3], [-1, 5, -1]], 4, source="r", apply_reranker_stage=True)

    baseline_cells = {(c["row"], c["col"]) for c in baseline["candidate_cells"]}
    reranked_cells = {(c["row"], c["col"]) for c in reranked["candidate_cells"]}
    assert baseline_cells == reranked_cells
