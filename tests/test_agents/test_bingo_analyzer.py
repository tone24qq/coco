from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from agent import PREDICT_REQUIRED_MESSAGE, BacktestRequest, BingoAnalyzer, app


def _make_recent(start_issue: int = 1, periods: int = 10) -> list[dict]:
    recent = []
    base_numbers = list(range(1, 21))
    for i in range(periods):
        shift = i % 60
        numbers = [((n + shift - 1) % 80) + 1 for n in base_numbers]
        recent.append({"issue": start_issue + i, "numbers": numbers})
    return recent


def _make_recent_from_csv(
    analyzer: BingoAnalyzer, start_issue: int, periods: int
) -> list[dict]:
    rows = analyzer.df[
        analyzer.df["期別"].between(start_issue, start_issue + periods - 1)
    ]
    assert len(rows) == periods
    draws = []
    for _, row in rows.iterrows():
        draws.append(
            {
                "issue": int(row["期別"]),
                "numbers": sorted(
                    analyzer.draw_numbers[analyzer.issue_to_index[int(row["期別"])]]
                ),
            }
        )
    return draws


def _build_small_csv(path: Path, draws: int = 36) -> None:
    rows = []
    for i in range(draws):
        start = i % 80
        nums = sorted([((start + k) % 80) + 1 for k in range(20)])
        row = {"期別": 300000000 + i}
        for j, n in enumerate(nums, start=1):
            row[f"獎號{j}"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


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
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000001, periods=14)
    recent_draws = [item["numbers"] for item in recent_payload]
    pred = analyzer.predict_next(recent_draws=recent_draws, latest_issue=115000014)
    assert pred["short_window"] == 14
    assert pred["latest_issue"] == 115000014
    assert pred["target_issue"] == 115000015
    assert pred["board_type"] in {
        "爆發盤",
        "雙區震盪盤",
        "均衡盤",
        "修正盤",
        "中段主導盤",
    }
    assert sum(pred["predicted_zone_counts"].values()) == 20
    assert len(pred["predicted_numbers_top20"]) == 20
    assert len(set(pred["predicted_numbers_top20"])) == 20
    assert len(pred["top3_triplet"]["numbers"]) == 3
    assert len(set(pred["top3_triplet"]["numbers"])) == 3
    assert all(1 <= n <= 80 for n in pred["top3_triplet"]["numbers"])
    explain = pred["top3_triplet"]["explain"]
    assert explain["weights"] == {
        "recent_weight": 0.2,
        "history_similar_weight": 0.5,
        "other_weight": 0.3,
    }
    assert len(explain["similar_cases_top10"]) == 10
    assert explain["similar_cases_used"] >= 10
    assert len(explain["number_contributions"]) == 3


def test_predict_changes_when_history_csv_is_empty(tmp_path: Path) -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000001, periods=14)
    recent_draws = [item["numbers"] for item in recent_payload]

    with_history = analyzer.predict_next(
        recent_draws=recent_draws, latest_issue=115000014
    )

    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=["期別"] + [f"獎號{i}" for i in range(1, 21)]).to_csv(
        empty_csv, index=False
    )
    empty_analyzer = BingoAnalyzer(csv_path=empty_csv)
    without_history = empty_analyzer.predict_next(
        recent_draws=recent_draws, latest_issue=115000014
    )

    assert (
        with_history["top3_triplet"]["numbers"]
        != without_history["top3_triplet"]["numbers"]
    )
    assert with_history["top3_triplet"]["explain"]["similar_cases_used"] > 0
    assert without_history["top3_triplet"]["explain"]["similar_cases_used"] == 0


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

    valid_payload = {
        "recent": _make_recent(start_issue=115012545, periods=20),
        "top_k": 20,
    }
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["short_window"] == 20
    assert data["latest_issue"] == 115012564
    assert data["target_issue"] == 115012565
    assert len(data["predicted_numbers_top20"]) == 20
    assert len(data["top3_triplet"]["numbers"]) == 3


def test_fastapi_analysis_and_health() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200

    analysis = client.get("/analysis")
    assert analysis.status_code == 200
    data = analysis.json()
    assert "basic" in data and "dynamic" in data


def test_run_top3_backtest_exports_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "small.csv"
    _build_small_csv(csv_path)
    analyzer = BingoAnalyzer(csv_path=csv_path)
    result = analyzer.run_top3_backtest(
        request=BacktestRequest(
            windows=[20],
            alphas=[0.95],
            lambdas=[1.0],
            recent_n=20,
            candidate_pool_size=12,
            output_dir=str(tmp_path / "out"),
        )
    )

    outputs = result["output_files"]
    for key in ["backtest_detail", "experiments", "best_config", "report"]:
        assert Path(outputs[key]).exists()

    detail_df = pd.read_csv(outputs["backtest_detail"])
    assert {
        "issue",
        "method",
        "P_t",
        "Y_t",
        "hit_count_t",
        "triple_hit_t",
        "precision_at_3",
        "recall_at_3",
    }.issubset(detail_df.columns)
