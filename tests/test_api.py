from typing import Any, Dict
import math


def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "OK"


def test_predict_valid_small(client, make_grid):
    grid = make_grid(4, 4)
    payload: Dict[str, Any] = {"grid": grid, "target_num": 6, "iterations": 32}
    resp = client.post("/predict", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert len(body["predictions"]) > 0
    assert isinstance(body["full_probabilities"], dict)


def test_predict_duplicate_numbers(client):
    bad = [[1, 1], [2, -1]]
    resp = client.post("/predict", json={"grid": bad})
    assert resp.status_code == 500
    assert "duplicate" in resp.json()["detail"].lower()


def test_predict_too_small(client):
    resp = client.post("/predict", json={"grid": [[-1]]})
    assert resp.status_code == 500
    assert "at least 2x2" in resp.json()["detail"].lower()


def test_heatmap_endpoint(client):
    grid = [[1, -1], [2, -1]]
    resp = client.post("/heatmap", json={"grid": grid, "k": 3, "iterations": 8})
    body = resp.json()
    assert resp.status_code == 200
    assert "heatmap" in body
    assert body["heatmap"].startswith("iVBOR")


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


def is_illegal_float(val):
    return isinstance(val, float) and (math.isnan(val) or math.isinf(val))


def test_result_top_k_all(client, make_grid):
    grid4x4 = [[-1 for _ in range(2)] for _ in range(2)]
    payload = {"grid": grid4x4, "result_top_k": 0, "iterations": 1}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    body = resp.json()

    # 🛡 檢查 predictions 裡是否含非法浮點
    for i, pred in enumerate(body.get("predictions", [])):
        for k, v in pred.items():
            if is_illegal_float(v):
                raise ValueError(f"🚨 非法浮點數出現在 predictions[{i}]['{k}'] = {v}")

    # 🛡 檢查 full_probabilities
    for pos, probs in body.get("full_probabilities", {}).items():
        for num_str, prob in probs.items():
            if is_illegal_float(prob):
                raise ValueError(f"🚨 非法浮點數出現在 full_probabilities[{pos}]['{num_str}'] = {prob}")

    assert len(body["predictions"]) == 16


def test_unique_dispatch_none(client, make_grid):
    grid4x4 = [[-1 for _ in range(2)] for _ in range(2)]
    payload = {"grid": grid4x4, "target_num": None, "iterations": 1}
    resp = client.post("/predict", json=payload)
    body = resp.json()
    assert body["mode"] == "target"
    for probs in body["full_probabilities"].values():
        assert len(probs) == 1


def test_single_num_filter(client, make_grid):
    grid4x4 = [[-1 for _ in range(2)] for _ in range(2)]
    payload = {"grid": grid4x4, "target_num": 42, "result_top_k": 5, "iterations": 1}
    resp = client.post("/predict", json=payload)
    probs = resp.json()["full_probabilities"]
    for p in probs.values():
        assert set(p.keys()) == {"42"}