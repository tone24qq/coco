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
    assert isinstance(data, list)
    assert len(data) == 4
