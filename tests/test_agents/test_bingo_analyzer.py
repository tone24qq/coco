from fastapi.testclient import TestClient

from agent import MODEL_VERSION, PREDICT_REQUIRED_MESSAGE, BingoAnalyzer, app


def _make_recent(start_issue: int = 1, periods: int = 20) -> list[dict]:
    recent = []
    base = list(range(1, 21))
    for idx in range(periods):
        shift = (idx * 3) % 80
        numbers = [((n + shift - 1) % 80) + 1 for n in base]
        recent.append({"issue": start_issue + idx, "numbers": numbers})
    return recent


def _make_recent_from_history(analyzer: BingoAnalyzer, periods: int = 20) -> list[dict]:
    rows = analyzer.df.tail(periods)
    payload = []
    for _, row in rows.iterrows():
        issue = int(row["期別"])
        payload.append(
            {
                "issue": issue,
                "numbers": analyzer.draw_numbers[analyzer.issue_to_index[issue]],
            }
        )
    return payload


def test_predict_response_contract() -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_history(analyzer, periods=20)
    recent_draws = [x["numbers"] for x in recent_payload]
    latest_issue = recent_payload[-1]["issue"]

    result = analyzer.predict_next(recent_draws=recent_draws, latest_issue=latest_issue)
    assert result["prediction_period"] == latest_issue + 1
    assert len(result["ranked_numbers"]) == 80
    assert len(result["scores"]) == 80
    assert len(result["top20"]) == 20
    assert result["model_version"] == MODEL_VERSION
    assert len(result["data_hash"]) == 16
    assert set(result["ranked_numbers"]) == set(range(1, 81))


def test_feature_analysis_contains_required_blocks() -> None:
    analyzer = BingoAnalyzer()
    report = analyzer.feature_analysis()
    for key in [
        "consecutive",
        "tail_distribution",
        "small_big",
        "hot_cold",
        "inter_draw_diff",
        "zone_density",
        "cooccurrence_matrix",
    ]:
        assert key in report
    assert len(report["cooccurrence_matrix"]) == 80


def test_backtest_report_contains_v2_sections() -> None:
    analyzer = BingoAnalyzer()
    report = analyzer.full_report(
        recent_window=20, prediction_k=20, evaluation_window=60, gap=1, embargo=1
    )
    for key in [
        "main_backtest",
        "baselines",
        "feature_ablation",
        "gap_purge_embargo",
        "calibration",
        "proper_scoring",
        "shuffle_sanity_check",
        "dependency_ablation",
        "feature_stability",
    ]:
        assert key in report
    assert "brier_score" in report["proper_scoring"]


def test_fastapi_endpoints() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"] == MODEL_VERSION

    missing = client.post("/predict")
    assert missing.status_code == 400
    assert missing.json()["detail"] == PREDICT_REQUIRED_MESSAGE

    valid_payload = {
        "recent": _make_recent(start_issue=200000001, periods=20),
        "prediction_k": 20,
    }
    pred = client.post("/predict", json=valid_payload)
    assert pred.status_code == 200
    data = pred.json()
    assert len(data["ranked_numbers"]) == 80
    assert "confidence" in data

    bt = client.post(
        "/backtest",
        json={
            "recent_window": 20,
            "prediction_k": 20,
            "evaluation_window": 60,
            "gap": 1,
            "embargo": 1,
        },
    )
    assert bt.status_code == 200
    assert "main_backtest" in bt.json()
