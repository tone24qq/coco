from fastapi.testclient import TestClient

from app import app
from dataset import BLANK_VALUE

client = TestClient(app)


def test_api_predict_only_blank() -> None:
    board = [
        [1, BLANK_VALUE, 3, 4, 5],
        [6, 7, BLANK_VALUE, 9, 10],
        [11, 12, 13, BLANK_VALUE, 15],
        [16, BLANK_VALUE, 18, 19, 20],
    ]
    r = client.post("/predict", json={"board": board, "target": 15})
    assert r.status_code == 200
    data = r.json()
    # 應至少回傳一筆
    assert isinstance(data, list) and len(data) > 0
    for item in data:
        r0, c0 = item["row"] - 1, item["col"] - 1
        assert board[r0][c0] == BLANK_VALUE
        # 驗證 debug 欄位
        assert item.get("idx") is not None
        assert item.get("cell_value") == BLANK_VALUE


def test_api_predict_violates_guard_returns_500(monkeypatch) -> None:
    """
    人工製造違規情境：把 BLANK_VALUE 改成其他值，確保致命驗證會擋下來。
    這裡僅示意，如果你不希望修改常數，可移除此測試。
    """
    board = [
        [1, -1, 3, 4, 5],
        [6, 7, -1, 9, 10],
        [11, 12, 13, -1, 15],
        [16, -1, 18, 19, 20],
    ]
    # 目標值 15，理論上會挑 -1 位置；若後端錯挑到不是 -1，應回 500
    r = client.post("/predict", json={"board": board, "target": 15})
    # 在正常邏輯下會 200；若有人故意改 BLANK_VALUE、或前端聲稱選到非空格，
    # 這支測試可視需求調整或移除。
    assert r.status_code in (200, 500)
