from fastapi.testclient import TestClient

import src.api as api_module


class _StubPredictor:
    def predict_from_draws(self, df, min_history):
        if len(df) <= min_history:
            raise ValueError("not enough history for feature generation")
        return {
            "target_issue": int(df.iloc[-1]["issue"]) + 1,
            "top20_numbers": list(range(1, 21)),
            "compact10_numbers": list(range(1, 11)),
            "top3_core_group": [1, 2, 3],
            "score_table": [{"number": i, "score": 1.0 / i} for i in range(1, 81)],
            "board_type_prediction": "balanced",
        }


def _payload(periods: int):
    draws = []
    for i in range(periods):
        draws.append([((i + k) % 80) + 1 for k in range(20)])
    return {"recent_draws": draws}


def test_predict_requires_recent_draws(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)

    resp = client.post("/predict", json={})

    assert resp.status_code == 400
    assert "22–50" in resp.json()["detail"]


def test_predict_validates_shape_and_range(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)

    too_short = client.post("/predict", json=_payload(21))
    assert too_short.status_code == 400

    invalid_len = _payload(22)
    invalid_len["recent_draws"][3] = list(range(1, 20))
    resp_len = client.post("/predict", json=invalid_len)
    assert resp_len.status_code == 400
    assert "exactly 20" in resp_len.json()["detail"]

    duplicated = _payload(22)
    duplicated["recent_draws"][0] = [1] * 20
    resp_dup = client.post("/predict", json=duplicated)
    assert resp_dup.status_code == 400
    assert "duplicate" in resp_dup.json()["detail"]

    out_of_range = _payload(22)
    out_of_range["recent_draws"][1][0] = 81
    resp_oob = client.post("/predict", json=out_of_range)
    assert resp_oob.status_code == 400
    assert "out-of-range" in resp_oob.json()["detail"]


def test_predict_success_contains_analysis_report(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)

    resp = client.post("/predict", json=_payload(23))

    assert resp.status_code == 200
    body = resp.json()
    assert "analysis_report" in body
    assert "odd_even" in body["analysis_report"]
    assert "recent_frequency" in body["analysis_report"]


def test_predict_converts_value_error_to_400(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    monkeypatch.setitem(api_module.PREDICT_CFG, "feature_min_history", 100)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json=_payload(22))

    assert resp.status_code == 400
    assert resp.json()["detail"] == "not enough history for feature generation"
