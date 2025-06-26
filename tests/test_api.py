# tests/test_api.py
from typing import Any
import math


# ────────────────────────────── 共用工具 ──────────────────────────────
def is_illegal_float(val: Any) -> bool:
    """檢查值是否為 NaN / ±Inf（JSON 不允許）"""
    return isinstance(val, float) and (math.isnan(val) or math.isinf(val))


# ──────────────────────────────── 基本 API ────────────────────────────────
def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("status") == "OK"


def test_predict_valid_small(client, make_grid):
    grid = make_grid(4, 4)
    payload = {"grid": grid, "target_num": 6, "iterations": 32}
    resp = client.post("/predict", json=payload)
    body = resp.json()

    assert resp.status_code == 200
    assert body["predictions"]                     # 至少要有一筆
    assert isinstance(body["full_probabilities"], dict)

    # 應該不包含非法浮點
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


# ──────────────────────────────── Heatmap ────────────────────────────────
def test_heatmap_endpoint_base64(client):
    grid = [[1, -1], [2, -1]]
    resp = client.post("/heatmap", json={"grid": grid, "k": 3, "iterations": 8})
    body = resp.json()

    assert resp.status_code == 200
    assert body["heatmap"].startswith("iVBOR")     # PNG→Base64 標頭


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


# ──────────────────── 功能擴充：result_top_k / 唯一分派 ────────────────────
def test_result_top_k_all(client):
    # 4×4 全空盤 → 未知格 = 16 格
    grid4x4 = [[-1] * 4 for _ in range(4)]
    resp = client.post(
        "/predict",
        json={"grid": grid4x4, "result_top_k": 0, "iterations": 1},
    )
    body = resp.json()

    assert resp.status_code == 200
    assert len(body["predictions"]) == 16          # 全部都要回傳

    # 不得含非法浮點
    for i, pred in enumerate(body["predictions"]):
        for k, v in pred.items():
            assert not is_illegal_float(v), f"predictions[{i}]['{k}'] = {v} 非法浮點"

    for pos, probs in body["full_probabilities"].items():
        for num_str, prob in probs.items():
            assert not is_illegal_float(prob), f"full_probabilities[{pos}]['{num_str}'] = {prob} 非法浮點"


def test_unique_dispatch_none(client):
    grid4x4 = [[-1] * 4 for _ in range(4)]
    resp = client.post(
        "/predict",
        json={"grid": grid4x4, "target_num": None, "iterations": 1},
    )
    body = resp.json()

    assert resp.status_code == 200
    assert body["mode"] in ("unique", "target")    # 後端命名擇一
    for probs in body["full_probabilities"].values():
        assert len(probs) == 1                     # 每格只允許一個號碼


def test_single_num_filter(client):
    grid4x4 = [[-1] * 4 for _ in range(4)]
    resp = client.post(
        "/predict",
        json={"grid": grid4x4, "target_num": 42, "result_top_k": 5, "iterations": 1},
    )
    body = resp.json()
    probs = body["full_probabilities"]

    assert resp.status_code == 200
    assert all(set(p.keys()) == {"42"} for p in probs.values())