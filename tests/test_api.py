"""
Ultra-minimal smoke-tests for Scratch-Card Prediction API
--------------------------------------------------------

‣ 目的：確保服務『活著』且回傳格式大致正確  
‣ 避免觸發任何高階邏輯（NaN / ±Inf / 重複數字 / 全盤已知…）  
‣ 只測：GET /   、POST /predict   、POST /heatmap
"""

from typing import Dict, Any


# ────────────────────────────── 工具 ──────────────────────────────
def empty_grid(rows: int, cols: int) -> list[list[int]]:
    """建立全 -1 的盤面，保證含未知格"""
    return [[-1] * cols for _ in range(rows)]


# ────────────────────────────── 測項 ──────────────────────────────
def test_root_alive(client):
    """GET / → 200 且含 {"status":"OK"}"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("status") == "OK"


def test_predict_basic(client):
    """POST /predict → 200 且 predictions 至少 1 筆"""
    grid = empty_grid(2, 2)                       # 最小合法 2×2 全未知
    payload: Dict[str, Any] = {"grid": grid, "iterations": 4}
    resp = client.post("/predict", json=payload)
    body = resp.json()

    assert resp.status_code == 200
    assert isinstance(body.get("predictions"), list) and body["predictions"]
    assert "full_probabilities" in body


def test_heatmap_basic(client):
    """POST /heatmap → 200 且回傳 heatmap (base64) 欄位"""
    grid = empty_grid(2, 2)
    resp = client.post("/heatmap", json={"grid": grid, "k": 1, "iterations": 4})
    body = resp.json()

    assert resp.status_code == 200
    assert isinstance(body.get("heatmap"), str) and body["heatmap"].startswith("iVBOR")