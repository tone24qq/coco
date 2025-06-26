"""
End-to-end API tests for Scratch-Card Prediction Service
=======================================================

‣ 使用 FastAPI TestClient（由 pytest fixture `client` 提供）  
‣ 測基本健康、錯誤處理、Heatmap 端點，以及
  － result_top_k=0 ⇒ 回傳所有未知格
  － target_num=None ⇒ 唯一分派模式
  － target_num=42 ⇒ 濾單一號碼
‣ 嚴格檢查 NaN / ±Inf 不得出現在回傳 JSON
"""

from typing import Any
import math


# ────────────────────────────── 小工具 ──────────────────────────────
def is_illegal_float(val: Any) -> bool:
    """True ⇔ val 為 NaN / +Inf / -Inf（JSON 規範不允許）"""
    return isinstance(val, float) and (math.isnan(val) or math.isinf(val))


# ──────────────────────────────── 根目錄 ────────────────────────────────
def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("status") == "OK"


# ────────────────────────────── /predict ──────────────────────────────
def test_predict_valid_small(client, make_grid):
    """4×4 網格，目標號碼 6；只要回傳格式正確即可"""
    grid = make_grid(4, 4)
    payload = {"grid": grid, "target_num": 6, "iterations": 32}
    resp = client.post("/predict", json=payload)
    body = resp.json()

    assert resp.status_code == 200
    assert body["predictions"]                          # 有至少一筆
    assert isinstance(body["full_probabilities"], dict)

    # 機率值不得出現非法浮點
    for pred in body["predictions"]:
        assert not any(is_illegal_float(v) for v in pred.values())


def test_predict_duplicate_numbers(client):
    bad = [[1, 1], [2, -1]]
    resp = client.post("/predict", json={"grid": bad})
    assert resp.status_code == 500
    assert "duplicate" in resp.json()["detail"].lower()


def test_predict_too_small(client):
    resp = client.post("/predict", json={"grid": [[-1]]})
    assert resp.status_code == 500
    assert "at least 2x2" in resp.json()["detail"].lower()


# ────────────────────────────── /heatmap ──────────────────────────────
def test_heatmap_endpoint_base64(client):
    grid = [[1, -1], [2, -1]]
    resp = client.post("/heatmap", json={"grid": grid, "k": 3, "iterations": 8})
    body = resp.json()

    assert resp.status_code == 200
    assert body["heatmap"].startswith("iVBOR")          # PNG (base64) magic-bytes


def test_heatmap_endpoint_raw(client):
    grid = [[1, -1], [2, -1]]
    resp = client.post(
        "/heatmap",
        json={"grid": grid, "k": 3, "iterations": 8, "output_format": "raw"},
    )
    body = resp.json()

    assert resp.status_code == 200
    assert body["heatmap"] is None
    assert isinstance(body["prob_map"], list)


# ──────────────────────── 進階：result_top_k / unique ────────────────────────
def test_result_top_k_all(client):
    """result_top_k=0 ⇒ 應回傳「所有未知格」"""
    grid4x4 = [[-1] * 4 for _ in range(4)]
    resp = client.post(
        "/predict",
        json={"grid": grid4x4, "result_top_k": 0, "iterations": 1},
    )
    body = resp.json()

    assert resp.status_code == 200
    assert len(body["predictions"]) == 16               # 4×4 全空 ⇒ 16 格
    # 保證無 NaN / ±Inf
    for pred in body["predictions"]:
        assert not any(is_illegal_float(v) for v in pred.values())
    for probs in body["full_probabilities"].values():
        assert not any(is_illegal_float(p) for p in probs.values())


def test_unique_dispatch_none(client):
    """target_num=None ⇒ 唯一分派，每格只允許 1 個號碼"""
    grid4x4 = [[-1] * 4 for _ in range(4)]
    resp = client.post(
        "/predict",
        json={"grid": grid4x4, "target_num": None, "iterations": 1},
    )
    body = resp.json()

    assert resp.status_code == 200
    assert body["mode"] in ("unique", "target")         # 後端名字擇一
    for probs in body["full_probabilities"].values():
        assert len(probs) == 1


def test_single_num_filter(client):
    """target_num=42 ⇒ full_probabilities 只能出現 '42'"""
    grid4x4 = [[-1] * 4 for _ in range(4)]
    resp = client.post(
        "/predict",
        json={"grid": grid4x4, "target_num": 42, "result_top_k": 5, "iterations": 1},
    )
    body = resp.json()
    for probs in body["full_probabilities"].values():
        assert set(probs.keys()) == {"42"}