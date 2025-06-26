from typing import Any, Dict


def test_root_ok(client):
    """GET / → 200 & {"status":"OK"}"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("status") == "OK"


def test_predict_smoke(client, make_grid):
    """POST /predict → 至少回一筆預測，不炸就算過"""
    grid = make_grid(2, 2)                # 最小 2×2
    payload: Dict[str, Any] = {"grid": grid, "iterations": 4}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    # 只確認基本結構
    assert isinstance(body.get("predictions"), list) and body["predictions"]
    assert isinstance(body.get("full_probabilities"), (dict, list))