import importlib
import json

import numpy as np

import app as app_module
import model
from dataset import BLANK_VALUE


def test_predict_prefers_memory_cache(monkeypatch):
    monkeypatch.setenv("MEMORY_SAMPLE_LIMIT", "5")
    importlib.reload(model)
    importlib.reload(app_module)
    app_module._preload_memories()

    called = {"stream": False}

    def fake_stream(*args, **kwargs):  # pragma: no cover - patched in test
        called["stream"] = True
        return []

    monkeypatch.setattr(app_module, "memory_predict_stream", fake_stream)

    data = json.load(open("data_archives/4x5.json", "r", encoding="utf-8"))
    board = np.array(data[0]["board"], dtype=int)
    board[0, 0] = BLANK_VALUE
    target = data[0]["target"]

    payload = app_module.PredictRequest(board=board.tolist(), target=target)
    result = app_module.predict(payload)
    assert isinstance(result, list)
    assert called["stream"] is False
