from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_ragged_board_422() -> None:
    r = client.post("/predict", json={"board": [[1, 2, 3], [4, 5]], "target": 3})
    assert r.status_code == 422


def test_target_out_of_range_422() -> None:
    r = client.post("/predict", json={"board": [[-1, -1], [-1, -1]], "target": 99})
    assert r.status_code == 422
