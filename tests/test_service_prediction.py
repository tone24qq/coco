"""
服務整合測試：
用 FastAPI TestClient 打 /predict，
確認每種尺寸模型在雲端佈署路徑也能正常響應。
"""

import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from coco_service.main import app

MODELS_DIR = Path("models")
client = TestClient(app)


def dim_from_name(name: str) -> tuple[int, int]:
    m = re.search(r"(\d+)x(\d+)", name)
    return int(m.group(1)), int(m.group(2))


def make_payload(rows: int, cols: int) -> dict:
    board = (-np.ones((rows, cols), dtype=int)).tolist()
    board[0][0] = 1
    return {"board": board, "target": 1}


@pytest.mark.parametrize("model_path", MODELS_DIR.glob("*.pkl"))
def test_api_predict_endpoint(model_path: Path) -> None:
    rows, cols = dim_from_name(model_path.name)
    payload = make_payload(rows, cols)

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    for item in data:
        assert 0 <= item["row"] < rows
        assert 0 <= item["col"] < cols
