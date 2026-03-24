import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app.app)


def test_predict_smoke(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "latest_known_issue": "1001",
        "target_issue": "1002",
        "model_version": "small_transformer_v2",
        "feature_version": "rank_window_v2",
        "data_source": "mock",
        "fetch_attempts": [],
        "score_type": "ranking_score",
        "scores": [{"number": i, "score": float(i)} for i in range(1, 81)],
        "top20": [{"number": i, "score": float(i)} for i in range(80, 60, -1)],
        "top3": [
            {"number": 80, "score": 80.0},
            {"number": 79, "score": 79.0},
            {"number": 65, "score": 65.0},
        ],
        "diversity_relaxed": False,
        "stale_issues": 0,
        "is_stale": False,
        "drift_metadata": {
            "trained_up_to_issue": "1000",
            "baseline_metrics": {},
            "feature_version": "rank_window_v2",
            "expected_input_schema": [],
        },
    }
    monkeypatch.setattr("app.predict", lambda: payload)

    response = client.get("/predict")
    if response.status_code != 200 or len(response.json()["scores"]) != 80:
        pytest.fail("/predict smoke failed")


def test_predict_remote_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.predict", lambda: (_ for _ in ()).throw(RuntimeError("fetch failed"))
    )
    response = client.get("/predict")
    if response.status_code != 500:
        pytest.fail("expected 500 for fetch failure")
