from fastapi.testclient import TestClient

from agent import PREDICT_REQUIRED_MESSAGE, BingoAnalyzer, app


def _make_recent(start_issue: int = 1, periods: int = 10) -> list[dict]:
    recent = []
    base_numbers = list(range(1, 21))
    for i in range(periods):
        shift = i % 60
        numbers = [((n + shift - 1) % 80) + 1 for n in base_numbers]
        recent.append({"issue": start_issue + i, "numbers": numbers})
    return recent


def test_data_loaded_and_sorted() -> None:
    analyzer = BingoAnalyzer()
    assert not analyzer.df.empty
    issues = analyzer.df["期別"].tolist()
    assert issues == sorted(issues)
    assert analyzer.matrix.shape[1] == 80


def test_basic_statistics_structure() -> None:
    analyzer = BingoAnalyzer()
    stats = analyzer.basic_statistics(top_n_triplets=5)
    assert stats["total_draws"] > 0
    assert len(stats["number_total_counts"]) == 80
    assert len(stats["number_probabilities"]) == 80
    assert len(stats["top_triplets"]) == 5


def test_predict_next_output_constraints_with_recent() -> None:
    analyzer = BingoAnalyzer()
    recent_draws = [sorted(item["numbers"]) for item in _make_recent(periods=25)]
    pred = analyzer.predict_next(recent_draws=recent_draws)
    assert pred["short_window"] == 25
    assert pred["board_type"] in {
        "爆發盤",
        "雙區震盪盤",
        "均衡盤",
        "修正盤",
        "中段主導盤",
    }
    assert sum(pred["predicted_zone_counts"].values()) == 20
    assert len(pred["predicted_numbers"]) == 20
    assert len(set(pred["predicted_numbers"])) == 20
    assert len(pred["top_triplet_prediction"]) == 3


def test_fastapi_predict_requires_recent() -> None:
    client = TestClient(app)
    response = client.post("/predict")
    assert response.status_code == 400
    assert response.json()["detail"] == PREDICT_REQUIRED_MESSAGE


def test_fastapi_predict_validates_recent_and_predicts() -> None:
    client = TestClient(app)

    invalid = {"recent": _make_recent(periods=9)}
    invalid_resp = client.post("/predict", json=invalid)
    assert invalid_resp.status_code == 422

    valid_payload = {"recent": _make_recent(periods=20), "top_k": 20}
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["short_window"] == 20
    assert len(data["predicted_numbers"]) == 20


def test_fastapi_analysis_and_health() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200

    analysis = client.get("/analysis")
    assert analysis.status_code == 200
    data = analysis.json()
    assert "basic" in data and "dynamic" in data
