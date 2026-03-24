from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
from src.runtime_history import build_runtime_history


@pytest.fixture
def client() -> TestClient:
    return TestClient(app.app)


def test_predict_smoke(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "history.csv"
    rows = [
        {
            "issue": 1001,
            "draw_time": "2026-01-01T00:00:00",
            **{f"n{i}": i for i in range(1, 21)},
        },
        {
            "issue": 1002,
            "draw_time": "2026-01-01T00:05:00",
            **{f"n{i}": i + 1 for i in range(1, 21)},
        },
    ]

    import pandas as pd

    pd.DataFrame(rows).to_csv(input_path, index=False)
    runtime_dir = tmp_path / "runtime"
    build_runtime_history(input_path, runtime_dir)

    monkeypatch.setattr("src.inference.DEFAULT_RUNTIME_DIR", runtime_dir)

    response = client.get("/predict")
    if response.status_code != 200:
        pytest.fail(f"unexpected status code: {response.status_code}")

    body = response.json()
    if len(body["scores"]) != 80:
        pytest.fail("scores chain should have 80 entries")


def test_predict_missing_artifact_returns_500(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path / "missing_runtime"
    monkeypatch.setattr("src.inference.DEFAULT_RUNTIME_DIR", runtime_dir)

    response = client.get("/predict")
    if response.status_code != 500:
        pytest.fail(f"expected 500, got {response.status_code}")
