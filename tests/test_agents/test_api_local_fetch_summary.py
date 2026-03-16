from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_and_predict_contract() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data

    r2 = client.get("/analysis")
    assert r2.status_code == 200


def test_fetch_endpoints_exist() -> None:
    assert client.post("/fetch/history-backfill").status_code == 200
    assert client.post("/fetch/consensus-check").status_code == 200
