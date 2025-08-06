import asyncio
import importlib

import numpy as np

import app as appmod


class DummyModel:
    def __init__(self) -> None:
        self.rows = 2
        self.cols = 2
        self.num_fields = 4
        self.num_values = 4

    def __call__(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - simple stub
        batch, n = x.shape
        return np.zeros((batch, n, self.num_values))

    def eval(self) -> None:  # pragma: no cover - compatibility stub
        pass


def test_app_predict_stable_order() -> None:
    importlib.reload(appmod)
    appmod.models.clear()
    appmod.models[(2, 2)] = DummyModel()
    board = np.full((2, 2), appmod.BLANK_VALUE).tolist()
    req = appmod.PredictRequest(board=board, target=1)
    res = asyncio.run(appmod.predict(req))
    coords = [(p.row, p.col) for p in res]
    assert coords == [(1, 1), (1, 2), (2, 1)]
