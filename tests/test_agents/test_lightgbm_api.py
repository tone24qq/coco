from fastapi.testclient import TestClient

import src.api as api_module


class _StubPredictor:
    feature_version = "v3_core20"
    runtime_config = {
        "core_windows": {"freq_long": 20, "pmi_window": 20, "handoff_window": 20}
    }

    def __init__(self):
        self.last_min_history = None

    def predict_from_draws(self, df, min_history):
        self.last_min_history = min_history
        if len(df) <= min_history:
            raise ValueError("not enough history for feature generation")
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
            "history_length_used": len(df),
            "feature_mode": "short",
            "degraded_features": [],
            "effective_windows": {
                "freq_long": min(len(df), 20),
                "pmi_window": min(len(df), 20),
                "handoff_window": min(len(df), 20),
            },
        }


def _payload(periods: int):
    draws = []
    for i in range(periods):
        draws.append([((i + k) % 80) + 1 for k in range(20)])
    return {"recent_draws": draws}


def test_predict_auto_fetch_when_recent_draws_missing(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    monkeypatch.setattr(
        api_module,
        "build_recent_draws",
        lambda fetcher, min_draws, max_draws: (
            _payload(23)["recent_draws"],
            [
                type(
                    "R",
                    (),
                    {"issue": 2000 + i, "draw_time": "2026-01-01", "numbers": draw},
                )
                for i, draw in enumerate(_payload(23)["recent_draws"])
            ],
            "https://primary.example",
            [
                type(
                    "A",
                    (),
                    {
                        "source": "https://winwin.tw/Bingo",
                        "ok": False,
                        "error": "timeout",
                    },
                )(),
                type(
                    "A",
                    (),
                    {"source": "https://primary.example", "ok": True, "error": None},
                )(),
            ],
        ),
    )
    client = TestClient(api_module.app)

    resp = client.post("/predict", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["auto_fetched"] is True
    assert body["data_source"] == "https://primary.example"
    assert body["recent_draws_count"] == 23
    assert body["first_issue_used"] == 2000
    assert body["last_issue_used"] == 2022
    assert body["issues_used"] == list(range(2000, 2023))
    assert body["history_length_used"] == 23
    assert body["fetch_attempts"] == [
        {"source": "https://winwin.tw/Bingo", "ok": False, "error": "timeout"},
        {"source": "https://primary.example", "ok": True, "error": None},
    ]
    assert "feature_mode" in body
    assert "degraded_features" in body
    assert "effective_windows" in body


def test_predict_auto_fetch_forces_single_source_when_configured(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    monkeypatch.setattr(api_module, "FORCE_FETCH_SOURCE", "https://winwin.tw/Bingo")

    captured = {}

    def _stub_build_recent_draws(fetcher, min_draws, max_draws):
        captured["sources"] = fetcher.sources
        return (
            _payload(23)["recent_draws"],
            [
                type(
                    "R",
                    (),
                    {"issue": 3000 + i, "draw_time": "2026-01-01", "numbers": draw},
                )
                for i, draw in enumerate(_payload(23)["recent_draws"])
            ],
            "https://winwin.tw/Bingo",
            [
                type(
                    "A",
                    (),
                    {
                        "source": "https://winwin.tw/Bingo",
                        "ok": True,
                        "error": None,
                    },
                )()
            ],
        )

    monkeypatch.setattr(api_module, "build_recent_draws", _stub_build_recent_draws)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json={})

    assert resp.status_code == 200
    assert captured["sources"] == ["https://winwin.tw/Bingo"]


def test_predict_auto_fetch_force_source_failure_does_not_fallback(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    monkeypatch.setattr(api_module, "FORCE_FETCH_SOURCE", "https://winwin.tw/Bingo")

    def _raise(*_args, **_kwargs):
        raise api_module.FetchDrawsError(
            "all sources failed: https://winwin.tw/Bingo: timeout"
        )

    monkeypatch.setattr(api_module, "build_recent_draws", _raise)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json={})

    assert resp.status_code == 502
    assert "https://winwin.tw/Bingo" in resp.json()["detail"]


def test_predict_validates_shape_and_range(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)

    too_short = client.post("/predict", json=_payload(0))
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
    assert body["model"] == "catboost"
    assert "analysis_report" in body
    assert "odd_even" in body["analysis_report"]
    assert "recent_frequency" in body["analysis_report"]
    assert "model_version" in body
    assert "feature_version" in body
    assert "training_data_snapshot" in body
    assert "raw_score_table" in body
    assert "calibrated_probability_table" in body
    assert len(body["top20_numbers"]) == 20
    assert len(body["top10_numbers"]) == 10
    assert len(body["top3_numbers"]) == 3
    assert len(body["top20_scores"]) == 20
    assert "size_summary" in body
    assert "odd_even_summary" in body
    assert body["auto_fetched"] is False
    assert body["data_source"] == "manual"
    assert body["first_issue_used"] is None
    assert body["last_issue_used"] is None
    assert body["issues_used"] == [None for _ in range(23)]
    assert body["fetch_attempts"] == []


def test_predict_converts_value_error_to_400(monkeypatch):
    predictor = _StubPredictor()
    monkeypatch.setattr(api_module, "PREDICTOR", predictor)
    monkeypatch.setitem(api_module.PREDICT_CFG, "feature_min_history", 100)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json=_payload(22))

    assert resp.status_code == 200
    assert predictor.last_min_history == 21


def test_predict_auto_fetch_error_to_502(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())

    def _raise(*_args, **_kwargs):
        raise api_module.FetchDrawsError("parse failed")

    monkeypatch.setattr(api_module, "build_recent_draws", _raise)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json={})

    assert resp.status_code == 502
    assert "auto fetch failed" in resp.json()["detail"]


def test_analysis_declares_recent_draws_optional(monkeypatch):
    monkeypatch.setattr(api_module, "PREDICTOR", _StubPredictor())
    client = TestClient(api_module.app)

    resp = client.get("/analysis")

    assert resp.status_code == 200
    assert resp.json()["recent_draws_rules"]["required"] is False
    assert resp.json()["recent_draws_rules"]["min"] >= 1
    assert resp.json()["recent_draws_rules"]["max"] >= 999


def test_predict_rejects_when_runtime_required_history_is_higher(monkeypatch):
    predictor = _StubPredictor()
    predictor.feature_version = "v3_core20"
    predictor.runtime_config = {
        "core_windows": {"freq_long": 200, "pmi_window": 200, "handoff_window": 200}
    }
    monkeypatch.setattr(api_module, "PREDICTOR", predictor)
    client = TestClient(api_module.app)

    resp = client.post("/predict", json=_payload(50))

    assert resp.status_code == 200
