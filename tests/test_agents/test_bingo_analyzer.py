import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from agent import (
    CSV_FILES,
    DEFAULT_LAST_DRAW_MAX_IN_TOPK,
    DEFAULT_LAST_DRAW_PENALTY,
    PREDICT_REQUIRED_MESSAGE,
    BacktestRequest,
    BingoAnalyzer,
    ScoreWeights,
    app,
)
from backtest import run_grid_search


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
        analyzer.df["issue"].between(start_issue, start_issue + periods - 1)
    ]
    assert len(rows) == periods
    draws = []
    for _, row in rows.iterrows():
        issue = int(row["issue"])
        draws.append(
            {
                "issue": issue,
                "numbers": sorted(
                    analyzer.draw_numbers[analyzer.issue_to_index[issue]]
                ),
            }
        )
    return draws


def _build_small_csv(path: Path, draws: int = 36) -> None:
    rows = []
    for i in range(draws):
        start = i % 80
        nums = sorted([((start + k) % 80) + 1 for k in range(20)])
        row = {"issue": 300000000 + i}
        for j, n in enumerate(nums, start=1):
            row[f"n{j}"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_data_loaded_and_sorted() -> None:
    analyzer = BingoAnalyzer()
    assert not analyzer.df.empty
    issues = analyzer.df["issue"].tolist()
    assert issues == sorted(issues)
    assert analyzer.matrix.shape[1] == 80


def test_default_loader_combines_2023_to_2026() -> None:
    analyzer = BingoAnalyzer()
    expected_rows = 0
    for csv_name in CSV_FILES:
        df = pd.read_csv(Path(csv_name))
        issue_col = "issue" if "issue" in df.columns else "期別"
        expected_rows += pd.to_numeric(df[issue_col], errors="coerce").notna().sum()
    assert len(analyzer.df) == expected_rows


def test_csv_loader_supports_three_formats(tmp_path: Path) -> None:
    issue_n_path = tmp_path / "issue_n.csv"
    pd.DataFrame(
        [
            {"issue": 1, **{f"n{i}": i for i in range(1, 21)}},
            {"issue": 2, **{f"n{i}": i + 1 for i in range(1, 21)}},
        ]
    ).to_csv(issue_n_path, index=False)
    assert len(BingoAnalyzer(csv_path=issue_n_path).draw_numbers) == 2

    zhong_path = tmp_path / "zhong.csv"
    pd.DataFrame(
        [
            {"期別": 1, **{f"獎號{i}": i for i in range(1, 21)}},
            {"期別": 2, **{f"獎號{i}": i + 1 for i in range(1, 21)}},
        ]
    ).to_csv(zhong_path, index=False)
    assert len(BingoAnalyzer(csv_path=zhong_path).draw_numbers) == 2

    onehot_path = tmp_path / "onehot.csv"
    rows = []
    for issue in [1, 2]:
        flags = {str(i): 1 if i <= 20 else 0 for i in range(1, 81)}
        rows.append({"issue": issue, **flags})
    pd.DataFrame(rows).to_csv(onehot_path, index=False)
    assert len(BingoAnalyzer(csv_path=onehot_path).draw_numbers) == 2


def test_basic_statistics_structure() -> None:
    analyzer = BingoAnalyzer()
    stats = analyzer.basic_statistics(top_n_triplets=5)
    assert stats["total_draws"] > 0
    assert len(stats["number_total_counts"]) == 80
    assert len(stats["number_probabilities"]) == 80
    assert len(stats["top_triplets"]) == 5
    assert "big_mid_small_stats" in stats


def test_predict_next_output_constraints_with_recent() -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000001, periods=14)
    recent_draws = [item["numbers"] for item in recent_payload]
    pred = analyzer.predict_next(recent_draws=recent_draws, latest_issue=115000014)
    assert pred["short_window"] == 14
    assert pred["latest_issue"] == 115000014
    assert pred["target_issue"] == 115000015
    assert sum(pred["predicted_zone_distribution"].values()) == 20
    assert sum(pred["predicted_big_mid_small_distribution"].values()) == 20
    assert len(pred["predicted_numbers_top20"]) == 20
    assert len(set(pred["predicted_numbers_top20"])) == 20
    assert "cluster_groups" in pred
    assert isinstance(pred["cluster_groups"], list)
    assert len(pred["top3_triplet"]["numbers"]) == 3
    assert len(pred["top_3_same_draw_combinations"]) == 3
    assert len(pred["top_10_candidate_numbers"]) == 10


def test_predict_next_penalizes_last_draw_overlap() -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000031, periods=12)
    recent_draws = [item["numbers"] for item in recent_payload]
    pred = analyzer.predict_next(recent_draws=recent_draws, latest_issue=115000042)

    last_draw_set = set(recent_draws[-1])
    overlap_count = len(set(pred["predicted_numbers_top20"]) & last_draw_set)

    assert overlap_count <= DEFAULT_LAST_DRAW_MAX_IN_TOPK
    assert (
        pred["explanation_of_influential_patterns"]["last_draw_penalty"]
        == DEFAULT_LAST_DRAW_PENALTY
    )
    assert (
        pred["explanation_of_influential_patterns"]["last_draw_max_in_topk"]
        == DEFAULT_LAST_DRAW_MAX_IN_TOPK
    )
    assert (
        pred["explanation_of_influential_patterns"]["last_draw_overlap_in_prediction"]
        == overlap_count
    )


def test_history_similarity_uses_sequence_window() -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000031, periods=12)
    recent_draws = [item["numbers"] for item in recent_payload]

    _, details = analyzer._history_pattern_similarity_component(
        recent_draws=recent_draws,
        latest_issue=115000042,
        sequence_window_size=8,
        top_n=5,
    )

    assert len(details) <= 5
    if details:
        first = details[0]
        assert first["sequence_window_size"] == 8
        assert first["sequence_end_issue"] >= first["sequence_start_issue"]
        assert first["next_issue"] > first["sequence_end_issue"]


def test_historical_data_verification_is_used() -> None:
    analyzer = BingoAnalyzer()
    recent_draws = [x["numbers"] for x in _make_recent(periods=20)]
    pred = analyzer.predict_next(recent_draws=recent_draws, latest_issue=999)
    hist = pred["history_verification"]
    assert hist["loaded_draws"] > 0
    assert hist["history_baseline"] >= 0
    assert pred["explanation_of_influential_patterns"]["similar_cases_used"] >= 0
    assert (
        pred["explanation_of_influential_patterns"]["sequence_similarity_window_size"]
        == 10
    )


def test_adaptive_weights_normalized() -> None:
    analyzer = BingoAnalyzer()
    weights = analyzer._adaptive_weights(
        {"zone_burst": True, "tail_cluster": True, "consecutive_spike": True}
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    base = ScoreWeights().as_dict()
    assert weights["zone_distribution"] > base["zone_distribution"]
    assert "cluster_pattern" in weights


def test_cluster_burst_analysis_detects_interval_tail_and_consecutive() -> None:
    analyzer = BingoAnalyzer()
    draws = [
        [1, 2, 3, 4, 5, 6, 7, 8, 25, 35, 45, 55, 65, 75, 10, 20, 30, 40, 50, 60],
        [21, 22, 23, 24, 25, 26, 27, 28, 18, 38, 48, 58, 68, 78, 9, 19, 29, 39, 49, 59],
    ]
    component, groups, meta = analyzer._cluster_burst_analysis(draws, window=2)

    assert len(component) == 80
    assert component.max() == 1.0
    assert meta["interval_cluster"] > 0
    assert meta["tail_cluster"] > 0
    assert meta["consecutive_cluster"] > 0
    assert any(len(group) >= 3 for group in groups)


def test_dynamic_cluster_weight() -> None:
    analyzer = BingoAnalyzer()
    spikes = {
        "zone_burst": False,
        "tail_cluster": False,
        "consecutive_spike": False,
        "cluster_burst": True,
    }
    weights_low = analyzer._adaptive_weights(spikes, cluster_score=1.0)
    weights_high = analyzer._adaptive_weights(spikes, cluster_score=9.0)

    assert abs(sum(weights_low.values()) - 1.0) < 1e-9
    assert abs(sum(weights_high.values()) - 1.0) < 1e-9
    assert weights_high["cluster_pattern"] > weights_low["cluster_pattern"]


def test_tail_blend() -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000021, periods=12)
    recent_draws = [item["numbers"] for item in recent_payload]

    component = analyzer._blended_tail_component(recent_draws)
    assert len(component) == 80
    assert component.max() == 1.0
    assert component.min() >= 0.0


def test_feature_extraction_contains_required_modules() -> None:
    analyzer = BingoAnalyzer()
    recent_draws = [x["numbers"] for x in _make_recent(periods=20)]
    dynamic = analyzer.dynamic_analysis(recent_draws=recent_draws, latest_issue=20)
    rf = dynamic["recent_features"]
    assert "zone_mean" in rf
    assert "range_mean" in rf
    assert "tail" in rf
    assert "gaps" in rf
    assert "skip" in rf
    assert "consecutive" in rf


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
    assert len(data["predicted_numbers_top20"]) == 20
    assert "predicted_zone_distribution" in data
    assert "explanation_of_influential_patterns" in data


def test_run_top3_backtest_exports_files_and_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "small.csv"
    _build_small_csv(csv_path, draws=80)
    analyzer = BingoAnalyzer(csv_path=csv_path)
    outdir = tmp_path / "out"
    result = analyzer.run_top3_backtest(
        request=BacktestRequest(
            windows=[20],
            alphas=[0.95],
            lambdas=[1.0],
            recent_n=20,
            candidate_pool_size=12,
            random_runs=200,
            max_steps=30,
            output_dir=str(outdir),
        )
    )

    outputs = result["output_files"]
    for key in ["backtest_detail", "experiments", "best_config", "report"]:
        assert Path(outputs[key]).exists()

    detail_df = pd.read_csv(outputs["backtest_detail"])
    assert detail_df["issue"].nunique() == 30

    experiments_df = pd.read_csv(outputs["experiments"])
    random_row = experiments_df[experiments_df["method"] == "random"].iloc[0]
    assert random_row["random_runs"] == 200
    assert random_row["triple_hit_rate_std"] >= 0

    best_cfg = json.loads(Path(outputs["best_config"]).read_text(encoding="utf-8"))
    assert "best_overall" in best_cfg
    assert "best_recent" in best_cfg

    report = Path(outputs["report"]).read_text(encoding="utf-8")
    assert "best_overall" in report
    assert "best_recent" in report
    assert "±" in report


def test_grid_search_saves_best_params(tmp_path: Path) -> None:
    csv_path = tmp_path / "small.csv"
    _build_small_csv(csv_path, draws=260)
    analyzer = BingoAnalyzer(csv_path=csv_path)
    out = tmp_path / "best_params.json"

    best = run_grid_search(analyzer, train_window=120, max_steps=20, output_path=out)
    assert out.exists()
    assert best["alpha"] in [0.7, 0.8, 0.9, 0.95]
    assert best["lambda"] in [0.3, 0.8, 1.5, 2.5]
    assert set(best["metrics"].keys()) == {
        "top20_hit_rate",
        "top10_hit_rate",
        "top3_hit_rate",
    }


def test_walk_forward_backtest_endpoint() -> None:
    client = TestClient(app)
    resp = client.post(
        "/backtest/walk-forward", json={"train_window": 100, "max_steps": 10}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["steps"] == 10
    assert "metrics" in data
    assert "avg_top10_hits" in data["metrics"]


def test_predict_top3_endpoint(tmp_path: Path) -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    best_config_path = artifacts / "best_config.json"
    best_config_path.write_text(
        json.dumps(
            {
                "best_overall": {
                    "method": "hybrid",
                    "window": 10,
                    "alpha": 0.95,
                    "lambda": 1.0,
                    "overall_triple_hit_rate": 0.1,
                    "recent_triple_hit_rate": 0.2,
                },
                "best_recent": {
                    "method": "freq_only",
                    "window": 10,
                    "alpha": 0.95,
                    "lambda": 0.0,
                    "overall_triple_hit_rate": 0.1,
                    "recent_triple_hit_rate": 0.2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    client = TestClient(app)
    payload = {
        "recent": _make_recent(start_issue=101, periods=20),
        "window": 15,
        "alpha": 0.9,
        "lambda": 0.8,
        "candidate_pool_size": 10,
    }
    resp = client.post("/predict/top3", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["top3"]) == 3
    assert data["config_used"]["use"] == "recent"
    assert data["config_used"]["config"]["window"] == 15
    assert data["config_used"]["config"]["alpha"] == 0.9
    assert data["config_used"]["config"]["lambda"] == 0.8
    assert data["config_used"]["config"]["candidate_pool_size"] == 10
    assert "single_scores" in data["diagnostics"]
    assert "pair_score_sum" in data["diagnostics"]

    resp_overall = client.post("/predict/top3?use=overall", json=payload)
    assert resp_overall.status_code == 200
    assert resp_overall.json()["config_used"]["use"] == "overall"


def test_predict_top3_missing_best_config_returns_500() -> None:
    best_config_path = Path("artifacts") / "best_config.json"
    if best_config_path.exists():
        best_config_path.unlink()

    client = TestClient(app)
    payload = {"recent": _make_recent(start_issue=101, periods=20)}
    resp = client.post("/predict/top3", json=payload)
    assert resp.status_code == 500


def test_predict_top3_rejects_invalid_candidate_pool_size() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    (artifacts / "best_config.json").write_text(
        json.dumps(
            {
                "best_recent": {"window": 10, "alpha": 0.95, "lambda": 1.0},
                "best_overall": {"window": 10, "alpha": 0.95, "lambda": 1.0},
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(app)
    payload = {
        "recent": _make_recent(start_issue=101, periods=20),
        "candidate_pool_size": 5,
    }
    resp = client.post("/predict/top3", json=payload)
    assert resp.status_code == 422


def test_sequence_similarity_prediction_output_schema() -> None:
    analyzer = BingoAnalyzer()
    recent_payload = _make_recent_from_csv(analyzer, start_issue=115000021, periods=12)
    recent_draws = [item["numbers"] for item in recent_payload]
    pred = analyzer.predict_next_sequence_similarity(
        recent_draws=recent_draws,
        latest_issue=115000032,
        input_window_size=10,
        min_match_count=10,
        top_k=15,
        output_top_n=10,
    )

    assert pred["mode"] == "sequence_similarity_next_draw"
    assert pred["feature_version"].startswith("v2")
    assert pred["similarity_version"].startswith("v2")
    assert pred["adjustment_version"].startswith("v2")
    assert pred["input_window_size"] == 10
    assert pred["minimum_required_matches"] == 10
    assert len(pred["predicted_top_3"]) in [0, 3]
    if not pred.get("insufficient_matches"):
        assert len(pred["predicted_top_5"]) == 5
        assert len(pred["predicted_top_10"]) == 10
        assert pred["matched_sequence_count"] >= 10
        assert len(pred["top_similar_sequences"]) >= 10
        assert len(pred["top_number_scores"]) == 10
        assert "current_window_zone_counts" in pred["debug"]
        assert "trend_profile" in pred["debug"]
        assert "pattern_adjustment_detail" in pred["debug"]
        assert "top_similarity_component_breakdown" in pred["debug"]
        assert "prefilter_candidate_count" in pred
        assert "postfilter_candidate_count" in pred


def test_sequence_similarity_backtest_endpoint() -> None:
    client = TestClient(app)
    resp = client.post(
        "/backtest/sequence-similarity",
        json={
            "input_window_size": 10,
            "min_match_count": 10,
            "top_k": 15,
            "output_top_n": 10,
            "max_steps": 8,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "sequence_similarity_next_draw"
    assert data["steps"] == 8
    assert "top3_hit_rate" in data["metrics"]
    assert "ab_comparison" in data["metrics"]
    assert {"A", "B", "C", "D"}.issubset(data["metrics"]["ab_comparison"].keys())
    assert "sample_insufficient_rate" in data


def test_sequence_similarity_predict_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "recent": _make_recent(start_issue=5001, periods=10),
        "input_window_size": 10,
        "min_match_count": 10,
        "top_k": 15,
        "output_top_n": 10,
    }
    resp = client.post("/predict/sequence-similarity", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "sequence_similarity_next_draw"
    assert "prediction_basis_summary" in data
