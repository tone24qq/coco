from fastapi.testclient import TestClient

import src.api as api_module


class _StubPredictor:
    feature_version = "v3_core20"
    runtime_config = {
        "core_windows": {"freq_long": 200, "pmi_window": 200, "handoff_window": 200}
    }

    def __init__(self):
        self.last_min_history = None

    def predict_from_draws(self, df, min_history):
        self.last_min_history = min_history
        return {
            "model": "catboost",
            "target_issue": int(df.iloc[-1]["issue"]) + 1,
            "top20_numbers": list(range(1, 21)),
            "top10_numbers": list(range(1, 11)),
            "top3_numbers": [1, 2, 3],
            "top20_scores": {f"{i:02d}": 1.0 / i for i in range(1, 21)},
            "compact10_numbers": list(range(1, 11)),
            "top3_core_group": [1, 2, 3],
            "raw_score_table": [{"number": i, "score": 1.0 / i} for i in range(1, 81)],
            "calibrated_probability_table": [
                {"number": i, "probability": 1.0 / i} for i in range(1, 81)
            ],
            "score_table": [{"number": i, "score": 1.0 / i} for i in range(1, 81)],
            "board_type_prediction": "balanced",
            "big_count": 0,
            "small_count": 20,
            "size_summary": "大0 / 小20",
            "odd_count": 10,
            "even_count": 10,
            "odd_even_summary": "單10 / 雙10",
        }


def _payload(periods: int):
    return {
        "recent_draws": [
            [((i + k) % 80) + 1 for k in range(20)] for i in range(periods)
        ]
    }


def test_api_accepts_1_to_999_range_without_forced_201_history(monkeypatch):
    predictor = _StubPredictor()
    monkeypatch.setattr(api_module, "PREDICTOR", predictor)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json=_payload(50))
    assert resp.status_code == 200
    assert predictor.last_min_history == 49


def test_api_accepts_sufficient_runtime_history(monkeypatch):
    predictor = _StubPredictor()
    monkeypatch.setattr(api_module, "PREDICTOR", predictor)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json=_payload(220))
    assert resp.status_code == 200
    assert len(resp.json()["top20_numbers"]) == 20
    assert predictor.last_min_history == 201
