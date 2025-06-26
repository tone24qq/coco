# tests/test_api.py
from typing import Any, Dict


def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"


def test_predict_valid_small(client, make_grid):
    grid = make_grid(4, 4)
    payload: Dict[str, Any] = {"grid": grid, "target_num": 6, "iterations": 32}
    resp = client.post("/predict", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert len(body["predictions"]) > 0
    assert isinstance(body["full_probabilities"], dict)


def test_predict_duplicate_numbers(client):
    bad = [[1, 1], [2, -1]]
    resp = client.post("/predict", json={"grid": bad})
    assert resp.status_code == 500
    assert "duplicate" in resp.json()["detail"].lower()


def test_predict_too_small(client):
    resp = client.post("/predict", json={"grid": [[-1]]})
    assert resp.status_code == 500
    assert "at least 2x2" in resp.json()["detail"].lower()


def test_heatmap_endpoint(client):
    grid = [[1, -1], [2, -1]]
    resp = client.post("/heatmap", json={"grid": grid, "k": 3, "iterations": 8})
    body = resp.json()
    assert resp.status_code == 200
    assert "heatmap" in body
    assert body["heatmap"].startswith("iVBOR")


def test_heatmap_endpoint_raw(client):
    grid = [[1, -1], [2, -1]]
    resp = client.post(
        "/heatmap",
        json={"grid": grid, "k": 3, "iterations": 8, "output_format": "raw"},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["heatmap"] is None
    assert isinstance(body["prob_map"], list)
