import importlib

import numpy as np

import app as appmod


def test_predict_heatmap_fusion(monkeypatch):
    importlib.reload(appmod)
    appmod.models.clear()
    appmod.models[(2, 2)] = appmod.DynamicMET(4, 4)

    heatmap = np.array([[0.9, 0.1], [0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(appmod, "load_heatmap", lambda r, c, target=None: heatmap)

    board = [[-1, -1], [-1, -1]]
    payload = appmod.PredictRequest(board=board, target=1)
    res = appmod.predict(payload)
    assert res[0].row == 1 and res[0].col == 1
