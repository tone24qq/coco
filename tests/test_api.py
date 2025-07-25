from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_predict_endpoint():
    board = [
        [1, -1, -1, 4],
        [-1, 4, 1, -1],
        [-1, 1, 4, -1],
        [4, -1, -1, 1],
    ]
    response = client.post("/predict", json={"board": board, "target": 3})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 4


def test_hints_endpoint():
    board = [
        [1, -1, -1, 4],
        [-1, 4, 1, -1],
        [-1, 1, 4, -1],
        [4, -1, -1, 1],
    ]
    res = client.post("/hints", json={"board": board, "target": 3})
    assert res.status_code == 200
    data = res.json()
    assert "hints" in data
    assert isinstance(data["hints"], list)
