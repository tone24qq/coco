import asyncio
import importlib

import numpy as np

import app as appmod


def test_predict_only_blank_top3():
    importlib.reload(appmod)
    appmod.models.clear()
    appmod.models[(2, 3)] = appmod.DynamicMET(6, 6)
    board = np.array([[1, -1, 2], [-1, 3, 4]]).tolist()
    payload = appmod.BoardInput(board=board, target_value=1)
    result = asyncio.get_event_loop().run_until_complete(appmod.predict(payload))
    assert len(result) == 2
    blank_positions = {(0, 1), (1, 0)}
    assert all((item["row"], item["col"]) in blank_positions for item in result)
