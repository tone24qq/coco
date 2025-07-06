"""
Ultra-minimal smoke-tests for Scratch-Card Prediction API
---------------------------------------------------------

‣ 目的：確保服務『活著』且回傳格式大致正確
‣ 避免觸發任何高階邏輯（NaN / ±Inf / 重複數字 / 全盤已知…）
‣ 只測：GET /   、POST /predict   、POST /heatmap
"""

from typing import Any, Dict, List

import analyzer
from modules import generate_excel_style_card


# ────────────────────────────── 工具 ──────────────────────────────
def empty_grid(rows: int, cols: int) -> List[List[int]]:
    """建立全 -1 的盤面，保證含未知格"""
    return [[-1] * cols for _ in range(rows)]


# ────────────────────────────── 測項 ──────────────────────────────
def test_root_alive(client):
    """GET / → 200 且含 {"status":"OK"}"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("status") == "OK"


def test_predict_basic(client):
    """POST /predict → 200 且 predictions 至少 1 筆，full_probabilities 是 dict"""
    grid = empty_grid(2, 2)  # 最小合法 2×2 全未知
    payload: Dict[str, Any] = {
        "grid": grid,
        "target_num": 1,  # 指定一個號碼，避免走唯一分派
        "iterations": 4,
    }
    resp = client.post("/predict", json=payload)
    body = resp.json()

    assert resp.status_code == 200
    assert isinstance(body.get("predictions"), list) and len(body["predictions"]) > 0
    assert isinstance(body.get("top_predictions"), list)
    assert isinstance(body.get("full_probabilities"), dict)
    assert isinstance(body.get("final_recommendations"), list)
    if body["final_recommendations"]:
        assert "final_score" in body["final_recommendations"][0]


def test_heatmap_basic(client):
    """POST /heatmap → 200 且回傳 heatmap (base64) 欄位"""
    grid = empty_grid(2, 2)
    payload = {"grid": grid, "target_num": 1, "iterations": 4}  # heatmap 的目標號碼
    resp = client.post("/heatmap", json=payload)
    body = resp.json()

    assert resp.status_code == 200
    heat = body.get("heatmap")
    assert isinstance(heat, str)
    # PNG base64 一定會以 iVBOR 開頭
    assert heat.startswith("iVBOR")
    assert isinstance(body.get("predictions"), list)
    assert isinstance(body.get("top_predictions"), list)
    assert isinstance(body.get("full_probabilities"), dict)
    assert isinstance(body.get("final_recommendations"), list)
    if body["final_recommendations"]:
        assert "final_score" in body["final_recommendations"][0]


def test_heatmap_json(client):
    """POST /heatmap with output_format=json → prob_map JSON"""
    grid = empty_grid(2, 2)
    payload = {"grid": grid, "target_num": 1, "iterations": 4, "output_format": "json"}
    resp = client.post("/heatmap", json=payload)
    body = resp.json()

    assert resp.status_code == 200
    assert isinstance(body.get("prob_map"), list)
    assert body.get("heatmap") is None
    assert isinstance(body.get("predictions"), list)
    assert isinstance(body.get("top_predictions"), list)
    assert isinstance(body.get("full_probabilities"), dict)
    assert isinstance(body.get("final_recommendations"), list)


def test_debug_number_distribution(client, monkeypatch):
    def fake(*_a, **_k):
        return {1: {(0, 0): 2}}

    monkeypatch.setattr(analyzer, "compute_number_distribution", fake)
    resp = client.get("/debug/number_distribution?rows=2&cols=2")
    body = resp.json()

    assert resp.status_code == 200
    assert body["1"]["1,1"] == 2


def test_predict_1_based_top_left(client):
    grid = [[-1, 2], [3, 4]]
    payload = {"grid": grid, "target_num": 1, "iterations": 2}
    resp = client.post("/predict", json=payload)
    body = resp.json()
    pred = body["predictions"][0]
    assert pred["row"] == 1
    assert pred["col"] == 1


def test_predict_1_based_bottom_right(client):
    grid = [[1, 2], [3, -1]]
    payload = {"grid": grid, "target_num": 4, "iterations": 2}
    resp = client.post("/predict", json=payload)
    body = resp.json()
    pred = body["predictions"][0]
    assert pred["row"] == 2
    assert pred["col"] == 2


def test_predict_excludes_filled_cells(client):
    grid = [[1, -1], [-1, 4]]
    payload = {"grid": grid, "target_num": 3, "iterations": 2}
    resp = client.post("/predict", json=payload)
    body = resp.json()
    coords = {(p["row"], p["col"]) for p in body["predictions"]}
    assert (1, 1) not in coords
    assert (2, 2) not in coords


def test_predict_strategy_modern(client):
    grid = [[1, -1], [2, -1]]
    payload = {"grid": grid, "strategy": "modern"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert "final_recommendations" in resp.json()


def test_no_open_cells_in_top3(client):
    grid = generate_excel_style_card(8, 10).tolist()
    grid[0][0] = -1
    body = client.post("/predict", json={"grid": grid, "target_num": 1}).json()
    top3 = {(p["row"] - 1, p["col"] - 1) for p in body["top_predictions"]}
    assert all(grid[r][c] == -1 for r, c in top3)


def test_blank_types_predict_and_heatmap(client):
    grid = [[-1, 0], ["", -1]]
    resp_p = client.post("/predict", json={"grid": grid, "target_num": 1})
    assert resp_p.status_code == 200
    resp_h = client.post(
        "/heatmap",
        json={"grid": grid, "target_num": 1, "iterations": 4, "output_format": "json"},
    )
    assert resp_h.status_code == 200
