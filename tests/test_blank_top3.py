import importlib
import logging

import numpy as np

import app as appmod


def test_predict_only_blank_top3(caplog):
    importlib.reload(appmod)
    appmod.models.clear()
    appmod.models[(2, 3)] = appmod.DynamicMET(6, num_values=6, rows=2, cols=3)
    board = np.array([[1, -1, 2], [-1, 3, 4]]).tolist()
    payload = appmod.PredictRequest(board=board, target=1)
    with caplog.at_level(logging.INFO):
        result = appmod.predict(payload)
    assert len(result) == 2
    blank_positions = {(1, 2), (2, 1)}
    assert all((item.row, item.col) in blank_positions for item in result)
    assert any("預測機率" in rec.message for rec in caplog.records)
