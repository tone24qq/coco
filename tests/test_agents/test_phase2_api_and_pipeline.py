from pathlib import Path

from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app


def test_health_and_predict_schema_with_mocked_runtime(monkeypatch, synthetic_records) -> None:
    class FakeArtifacts:
        feature_columns = ["cand_hits_last_100"]
        metadata = {"created_at": "2026-01-01T00:00:00", "model_family": "test"}

    def fake_run_prediction(_artifacts, _cfg, _recent):
        return {
            "issue": "20260101099",
            "source": "manual",
            "dynamic_context_n": 30,
            "top20_numbers": list(range(1, 21)),
            "top10_numbers": list(range(1, 11)),
            "top3_numbers": [1, 4, 7],
            "top3_before_group_dedup": [1, 2, 3],
            "top3_after_group_dedup": [1, 4, 7],
            "retrieval_top_matches": [],
            "ranking_score_table": [],
            "metadata": {"runtime_history_issue_range": ["20260101098", "20260101098"]},
        }

    monkeypatch.setattr(
        api_module,
        "get_runtime",
        lambda: (
            FakeArtifacts(),
            {"history": {"min_dynamic_n": 20, "processed_path": "data/processed/history_processed.csv", "runtime_artifact_dir": "data/runtime_history"}, "auto_fetch": {"source": "x"}},
            None,
        ),
    )
    monkeypatch.setattr(api_module, "run_prediction", fake_run_prediction)

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert "processed_history_exists" in health.json()
    assert "compact_history_ready" in health.json()

    payload = {"recent_draws": [r.to_dict() for r in synthetic_records[-30:]]}
    pred = client.post("/predict", json=payload)
    assert pred.status_code == 200
    body = pred.json()
    assert sorted(body.keys()) == sorted(
        [
            "latest_fetched_issue",
            "target_issue",
            "top20_numbers",
            "big_count",
            "small_count",
            "odd_count",
            "even_count",
            "size_summary",
            "odd_even_summary",
        ]
    )
    assert len(body["top20_numbers"]) == 20
    assert body["big_count"] + body["small_count"] == 20
    assert body["odd_count"] + body["even_count"] == 20


def test_health_degraded_when_artifacts_missing(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_runtime",
        lambda: (
            None,
            {"history": {"min_dynamic_n": 20, "processed_path": "missing.csv", "runtime_artifact_dir": "data/runtime_history"}, "auto_fetch": {"sources": ["s"]}},
            "missing",
        ),
    )
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_get_runtime_normalizes_config_paths_to_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    api_module.get_runtime.cache_clear()
    _, cfg, _ = api_module.get_runtime()
    assert Path(cfg["models"]["dir"]).is_absolute()
    assert Path(cfg["history"]["processed_path"]).is_absolute()
    assert Path(cfg["provenance"]["audit_path"]).is_absolute()
    assert Path(cfg["snapshot"]["path"]).is_absolute()
