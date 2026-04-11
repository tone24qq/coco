from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_compact_target_response_schema_and_sorting() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 4})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"top10", "best_confidence_1_to_100"}
    assert len(payload["top10"]) <= 10
    confs = [float(item["confidence_1_to_100"]) for item in payload["top10"]]
    assert confs == sorted(confs, reverse=True)
    assert payload["best_confidence_1_to_100"] == payload["top10"][0]["confidence_1_to_100"]


def test_compact_multi_target_response_schema() -> None:
    board = [[1, -1], [-1, 4]]
    response = client.post("/infer_multi_target_positions", json={"board": board, "target_numbers": [2, 3]})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"top10", "best_confidence_1_to_100"}
    assert len(payload["top10"]) <= 10


def test_duplicate_opened_number_fail_fast() -> None:
    board = [[1, -1], [1, 4]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 2})
    assert response.status_code == 422


def test_already_opened_compact_response() -> None:
    board = [[1, 2], [3, -1]]
    response = client.post("/infer_target_position", json={"board": board, "target_number": 3})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"top10", "best_confidence_1_to_100"}
    assert payload["top10"][0]["row"] == 2
    assert payload["top10"][0]["col"] == 1
