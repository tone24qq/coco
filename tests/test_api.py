from fastapi.testclient import TestClient

from src.inference.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_version_endpoint():
    with TestClient(app) as client:
        resp = client.get("/version")
        assert resp.status_code == 200
        data = resp.json()
        assert {"git_sha", "vocab_size", "device"} <= data.keys()


def test_predict_validation_and_output():
    with TestClient(app) as client:
        # invalid value beyond range
        resp = client.post("/predict", json={"grid": [[0, 0], [0, 5]]})
        assert resp.status_code == 422

        # grid too large for model capacity
        big_grid = [[0] * 21 for _ in range(21)]
        resp = client.post("/predict", json={"grid": big_grid})
        assert resp.status_code == 400

        # float value
        resp = client.post("/predict", json={"grid": [[0.0, 0], [0, 0]]})
        assert resp.status_code == 422

        # string value
        resp = client.post("/predict", json={"grid": [["a", 0], [0, 0]]})
        assert resp.status_code == 422

        # valid grid
        resp = client.post("/predict", json={"grid": [[0, 0], [0, 0]]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["rows"] == 2 and data["cols"] == 2 and len(data["grid"]) == 2

        # alias "board" should behave the same as "grid"
        resp = client.post("/predict", json={"board": [[0, 0], [0, 0]]})
        assert resp.status_code == 200


def test_predict_with_target_topk():
    with TestClient(app) as client:
        resp = client.post(
            "/predict",
            json={"board": [[-1, -1], [-1, -1]], "target": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["target"] == 1
        preds = data["predictions"]
        assert len(preds) == 3
        for item in preds:
            assert {"row", "col", "prob"} <= item.keys()
        assert "[步驟1]" in data["log"]
