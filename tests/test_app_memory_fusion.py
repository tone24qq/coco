import json

import numpy as np
from fastapi.testclient import TestClient

from app import app  # noqa: E402
from dataset import BLANK_VALUE  # noqa: E402

client = TestClient(app)


def test_app_memory_fusion_top3() -> None:
    data = json.load(open("data_archives/4x5.json", "r", encoding="utf-8"))
    board = np.array(data[0]["board"], dtype=int)
    board[0, 1] = BLANK_VALUE
    board[2, 3] = BLANK_VALUE
    target = data[0]["target"]
    resp = client.post("/predict", json={"board": board.tolist(), "target": target})
    assert resp.status_code == 200
    res = resp.json()
    blank_count = np.sum(np.array(board) == BLANK_VALUE)
    assert len(res) == min(3, blank_count)
    scores = [item["score"] for item in res]
    assert scores == sorted(scores, reverse=True)
    for item in res:
        r0, c0 = item["row"] - 1, item["col"] - 1
        assert board[r0][c0] == BLANK_VALUE
