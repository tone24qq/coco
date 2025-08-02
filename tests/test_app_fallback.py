import importlib
import sys

import numpy as np

import app as appmod
import model


def test_app_predict_numpy(monkeypatch):
    orig_torch = sys.modules.get("torch")
    # Simulate environment without torch *before* importing modules that depend on it.
    monkeypatch.setitem(sys.modules, "torch", None)
    importlib.reload(model)
    importlib.reload(appmod)
    appmod.models.clear()
    appmod.models[(2, 2)] = model.DynamicMET(4, num_values=4, rows=2, cols=2)
    board = np.array([[1, 2], [3, -1]]).tolist()
    payload = appmod.PredictRequest(board=board, target_value=1)
    result = appmod.predict(payload)
    assert isinstance(result, list) and len(result) == 1
    for item in result:
        assert {"row", "col", "score"} <= set(item.model_dump().keys())
    monkeypatch.setitem(sys.modules, "torch", orig_torch)
    importlib.reload(model)
