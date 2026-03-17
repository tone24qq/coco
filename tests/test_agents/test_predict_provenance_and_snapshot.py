from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import src.api as api_module


class _StubPredictor:
    feature_version = "v3_core20"
    runtime_config = {
        "core_windows": {"freq_long": 200, "pmi_window": 200, "handoff_window": 200}
    }

    class _S:
        pipeline_version = "cascade_v1"
        version_id = "cascade_v1_flow"

    strategy = _S()

    def predict_from_draws(self, df, min_history):
        return {
            "model": "catboost",
            "target_issue": int(df.iloc[-1]["issue"]) + 1,
            "top20_numbers": list(range(1, 21)),
            "top10_numbers": list(range(1, 11)),
            "top3_numbers": [1, 2, 3],
            "top20_scores": {f"{i:02d}": 1.0 / i for i in range(1, 21)},
            "compact10_numbers": list(range(1, 11)),
            "top3_core_group": [1, 2, 3],
            "raw_score_table": [{"number": i, "score": 1.0 / i} for i in range(1, 81)],
            "ranking_score_table": [
                {"number": i, "score": 1.0 / i} for i in range(1, 81)
            ],
            "score_table": [{"number": i, "score": 1.0 / i} for i in range(1, 81)],
            "board_type_prediction": "balanced",
            "big_count": 10,
            "small_count": 10,
            "size_summary": "大10 / 小10",
            "odd_count": 10,
            "even_count": 10,
            "odd_even_summary": "單10 / 雙10",
            "history_length_used": len(df),
            "feature_mode": "short",
            "degraded_features": [],
            "effective_windows": {
                "freq_long": min(len(df), 200),
                "pmi_window": min(len(df), 200),
                "handoff_window": min(len(df), 200),
            },
        }


def _payload(periods: int):
    return {
        "recent_draws": [
            [((i + k) % 80) + 1 for k in range(20)] for i in range(periods)
        ]
    }


def test_predict_provenance_fields(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)
    resp = client.post("/predict", json=_payload(50))
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_source"] == "manual"
    assert body["recent_draws_count"] == 50
    assert body["history_length_used"] == 50
    assert "canonical_rows" in body
    assert "canonical_issue_start" in body
    assert "canonical_issue_end" in body
    assert "raw_manifest_file_count" in body
    assert "raw_manifest_total_rows" in body
    assert "coverage_year_start" in body
    assert "coverage_year_end" in body
    assert isinstance(body["detected_files"], list)
    assert "training_window_used" in body
    assert "analysis_engine" in body
    assert body["external_status"] == "not_requested"


def test_analysis_full_endpoint(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)
    resp = client.post("/analysis/full", json=_payload(25))
    assert resp.status_code == 200
    body = resp.json()
    assert "comprehensive" in body
    assert "shape_oe" in body
    assert body["summary"]["sample_size"] == 25


def test_history_snapshot_builder_outputs_files(tmp_path, monkeypatch):
    import pandas as pd

    from src.analysis.snapshots import build_history_snapshot

    rows = []
    for i in range(30):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 1000 + i,
                "draw_date": f"2026-01-{(i % 10) + 1:02d}",
                "numbers": json.dumps(nums, ensure_ascii=False),
            }
        )
    df = pd.DataFrame(rows)
    snapshot, meta = build_history_snapshot(df)
    assert not snapshot.empty
    assert meta["snapshot_type_counts"]["number"] == 80
    assert Path(meta["paths"]["history_snapshot"]).exists()
    assert Path(meta["paths"]["history_snapshot_meta"]).exists()
