import importlib
import json

import numpy as np

import app as app_module
import model
from dataset import BLANK_VALUE


def test_predict_uses_memory_cache(monkeypatch):
    importlib.reload(model)
    importlib.reload(app_module)

    called = {"cache": False}

    def fake_memory_predict(*args, **kwargs):  # pragma: no cover - patched in test
        called["cache"] = True
        return []

    monkeypatch.setattr(app_module, "memory_predict", fake_memory_predict)

    data = json.load(open("data_archives/4x5.json", "r", encoding="utf-8"))
    board = np.array(data[0]["board"], dtype=int)
    board[0, 0] = BLANK_VALUE
    target = data[0]["target"]

    payload = app_module.PredictRequest(board=board.tolist(), target=target)
    app_module.predict(payload)
    assert called["cache"] is True
