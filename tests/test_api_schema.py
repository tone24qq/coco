from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

BOARD = [
    [-1, -1, 16, -1, 12],
    [19, -1, 10, 5, 3],
    [20, -1, 6, 18, 1],
    [-1, 4, 13, -1, -1],
]


def _ok(resp):
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, list) and len(data) == 3
    assert {"row", "col", "score"} <= data[0].keys()


def test_predict_target_only():
    r = client.post("/predict", json={"board": BOARD, "target": 15})
    _ok(r)


def test_predict_target_value_only():
    r = client.post("/predict", json={"board": BOARD, "target_value": 15})
    _ok(r)


def test_predict_both_same():
    r = client.post(
        "/predict",
        json={"board": BOARD, "target": 15, "target_value": 15},
    )
    _ok(r)


def test_predict_both_conflict():
    r = client.post(
        "/predict",
        json={"board": BOARD, "target": 14, "target_value": 15},
    )
    assert r.status_code == 422
