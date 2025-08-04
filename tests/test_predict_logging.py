import json

import numpy as np
from fastapi.testclient import TestClient

from app import app
from dataset import BLANK_VALUE

client = TestClient(app)


def test_predict_logs(caplog):
    data = json.load(open("data_archives/4x5.json", "r", encoding="utf-8"))
    board = np.array(data[0]["board"], dtype=int)
    board[0, 1] = BLANK_VALUE
    board[2, 3] = BLANK_VALUE
    target = data[0]["target"]
    with caplog.at_level("INFO"):
        resp = client.post("/predict", json={"board": board.tolist(), "target": target})
    assert resp.status_code == 200
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "記憶庫" in messages
    assert "模型推理" in messages
    assert "記憶檢索" in messages
    assert "合併" in messages
