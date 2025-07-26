import asyncio
import importlib
import sys

import numpy as np

import app as appmod
import model


def test_app_predict_numpy(monkeypatch):
    orig_torch = sys.modules.get("torch")
    monkeypatch.setitem(sys.modules, "torch", None)
    model_no_torch = importlib.reload(model)
    monkeypatch.setattr(appmod, "torch", None, raising=False)
    monkeypatch.setattr(appmod, "DynamicMET", model_no_torch.DynamicMET, raising=False)
    appmod.models.clear()
    appmod.models[(2, 2)] = model_no_torch.DynamicMET(4, 5)
    board = np.array([[1, 2], [3, -1]]).tolist()
    payload = appmod.BoardInput(board=board, target_value=1)
    result = asyncio.get_event_loop().run_until_complete(appmod.predict(payload))
    assert isinstance(result, list) and len(result) == 3
    for item in result:
        assert {"row", "col", "score"} <= set(item.keys())
    monkeypatch.setitem(sys.modules, "torch", orig_torch)
    importlib.reload(model)
