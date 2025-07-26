import asyncio
import importlib
import sys
import types

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
    appmod.models[(2, 2)] = model.DynamicMET(4, 5)
    board = np.array([[1, 2], [3, -1]]).tolist()
    payload = types.SimpleNamespace(board=board, target_value=1)
    result = asyncio.get_event_loop().run_until_complete(appmod.predict(payload))
    assert isinstance(result, list) and len(result) == 1
    for item in result:
        assert {"row", "col", "score"} <= set(item.keys())
    monkeypatch.setitem(sys.modules, "torch", orig_torch)
    importlib.reload(model)
