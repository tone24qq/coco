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
            "ranking_score_table": [
                {
                    "number": i,
                    "rank_final": i,
                    "final_score": 1.0 / i,
                    "ranker_score": 1.0 / i,
                    "logistic_score": 1.0 / i,
                    "retrieval_score": 0.0,
                    "history_prior_score": 0.0,
                    "analysis_rerank_score": 0.0,
                    "local_peak_score": 0.0,
                }
                for i in range(1, 81)
            ],
            "metadata": {"feature_count": 1},
        }

    monkeypatch.setattr(api_module, "get_runtime", lambda: (FakeArtifacts(), {"history": {"min_dynamic_n": 20}, "auto_fetch": {"source": "x"}}, None))
    monkeypatch.setattr(api_module, "run_prediction", fake_run_prediction)

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    payload = {"recent_draws": [r.to_dict() for r in synthetic_records[-30:]]}
    pred = client.post("/predict", json=payload)
    assert pred.status_code == 200
    body = pred.json()
    assert len(body["ranking_score_table"]) == 80


def test_health_degraded_when_artifacts_missing(monkeypatch):
    monkeypatch.setattr(api_module, "get_runtime", lambda: (None, {"history": {"min_dynamic_n": 20}, "auto_fetch": {"sources": ["s"]}}, "missing"))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False
