import numpy as np
from fastapi.testclient import TestClient

import app as app_module
import neighbour_loader
from dataset import BLANK_VALUE

client = TestClient(app_module.app)


def test_load_nbr(tmp_path, monkeypatch):
    arr = np.zeros((5, 5), dtype=np.float32)
    np.save(tmp_path / "2x3_nbr_probs.npy", arr)
    monkeypatch.setattr(neighbour_loader, "NBR_DIR", tmp_path)
    neighbour_loader.NEIGHBOR_PROBS.clear()
    neighbour_loader.load_nbr(2, 3)
    assert (2, 3) in neighbour_loader.NEIGHBOR_PROBS


def test_app_neighbour_fusion(monkeypatch):
    board = np.array([[1, 2, 3], [4, BLANK_VALUE, BLANK_VALUE]])
    target = 1
    monkeypatch.setenv("NBR_ALPHA", "1")
    monkeypatch.setattr(app_module, "torch", None)

    class DummyModel:
        def __call__(self, inp):
            n = inp.shape[1]
            return np.zeros((1, n, 5))

    app_module.models[(2, 3)] = DummyModel()

    async def _noop(r, c):
        return None

    monkeypatch.setattr(app_module, "ensure_loaded", _noop)
    monkeypatch.setattr(app_module, "filter_by_target", lambda *a, **k: [])
    neighbour_loader.NEIGHBOR_PROBS[(2, 3)] = np.zeros((5, 5), dtype=np.float32)
    neighbour_loader.NEIGHBOR_PROBS[(2, 3)][1, 4] = 1.0
    resp = client.post("/predict", json={"board": board.tolist(), "target": target})
    assert resp.status_code == 200
    res = resp.json()
    assert res[0]["row"] == 2 and res[0]["col"] == 2
