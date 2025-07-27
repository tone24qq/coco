from fastapi.testclient import TestClient

from app import app
from dataset import BLANK_VALUE

client = TestClient(app)


def test_api_predict_only_blank():
    board = [
        [1, BLANK_VALUE, 3, 4, 5],
        [6, 7, BLANK_VALUE, 9, 10],
        [11, 12, 13, BLANK_VALUE, 15],
        [16, BLANK_VALUE, 18, 19, 20],
    ]
    response = client.post("/predict", json={"board": board, "target": 15})
    assert response.status_code == 200
    data = response.json()
    for item in data:
        r0, c0 = item["row"], item["col"]
        assert board[r0][c0] == BLANK_VALUE
