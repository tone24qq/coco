"""
核心單元測試：
確定 models/*.pkl 都可以被載入，且 predict_top_k() 不會丟例外，
並回傳符合 (row, col, score) 欄位的列表。
"""

import re
from pathlib import Path

import numpy as np
import pytest

from rf_infer.core import _load_model, predict_top_k

MODELS_DIR = Path("models")
TOP_K = 3


def find_models():
    return list(MODELS_DIR.glob("*.pkl"))


def dim_from_name(name: str) -> tuple[int, int]:
    m = re.search(r"(\d+)x(\d+)", name)
    if not m:
        raise ValueError(f"找不到尺寸資訊: {name}")
    return int(m.group(1)), int(m.group(2))


def dummy_board(rows: int, cols: int, target: int) -> np.ndarray:
    board = -np.ones((rows, cols), dtype=int)
    board[0, 0] = target
    return board


@pytest.mark.parametrize("model_path", find_models())
def test_model_can_predict(model_path: Path) -> None:
    rows, cols = dim_from_name(model_path.name)
    model = _load_model(str(model_path))

    target = 1
    board = dummy_board(rows, cols, target)

    result = predict_top_k(model, board, target, TOP_K)
    preds = result.get("predictions", [])

    assert isinstance(preds, list) and len(preds) <= TOP_K
    for item in preds:
        assert {"r", "c", "prob"} <= set(item)
        assert 0 <= item["r"] < rows
        assert 0 <= item["c"] < cols
        assert isinstance(item["prob"], (float, int))
