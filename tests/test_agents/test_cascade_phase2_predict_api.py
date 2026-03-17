from fastapi.testclient import TestClient

import src.api as api_module


class _StubCascadePredictor:
    feature_version = "v3_core20"
    runtime_config = {}

    def predict_from_draws(self, _df, min_history: int) -> dict:
        _ = min_history
        return {
            "model": "catboost_cascade",
            "strategy_version": "cascade_v1_flow",
            "target_issue": 1,
            "top20_numbers": list(range(1, 21)),
            "top10_numbers": list(range(1, 11)),
            "top3_numbers": [1, 2, 3],
            "top20_scores": {f"{i:02d}": 0.1 for i in range(1, 21)},
            "compact10_numbers": list(range(1, 11)),
            "top3_core_group": [1, 2, 3],
            "raw_score_table": [{"number": i, "score": 0.1} for i in range(1, 81)],
            "ranking_score_table": [{"number": i, "score": 0.1} for i in range(1, 81)],
            "score_table": [{"number": i, "score": 0.1} for i in range(1, 81)],
            "board_type_prediction": "balanced",
            "big_count": 10,
            "small_count": 10,
            "size_summary": "大10 / 小10",
            "odd_count": 10,
            "even_count": 10,
            "odd_even_summary": "單10 / 雙10",
            "history_length_used": 30,
            "feature_mode": "full",
            "degraded_features": [],
            "effective_windows": {},
            "cascade_debug": {"stage1_keep_count": 30, "stage2_keep_count": 10},
        }


def _payload() -> dict:
    draws = [[((i + k) % 80) + 1 for k in range(20)] for i in range(30)]
    return {"recent_draws": draws}


def test_api_hides_stage_debug_by_default(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "PREDICTOR", _StubCascadePredictor())
    c = TestClient(api_module.app)
    resp = c.post("/predict", json=_payload())
    assert resp.status_code == 200
    assert "cascade_debug" not in resp.json()


def test_api_includes_stage_debug_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "PREDICTOR", _StubCascadePredictor())
    c = TestClient(api_module.app)
    body = _payload()
    body["include_stage_details"] = True
    resp = c.post("/predict", json=body)
    assert resp.status_code == 200
    assert "cascade_debug" in resp.json()
