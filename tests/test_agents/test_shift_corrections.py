from __future__ import annotations

import json

import pandas as pd

from src import backtest
from src.backtest import _ablation_report_from_issue_rows, _build_error_shift_report
from src.predict import Predictor
from src.utils import MODELS_DIR, V3_CORE20_COLUMNS, apply_topk_group_dedup


class _DummyClassifier:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load_model(self, _path: str) -> None:
        self.loaded = True

    def predict_proba(self, x):
        import numpy as np

        vals = np.linspace(0.2, 0.8, len(x), dtype=float)
        return np.vstack([1.0 - vals, vals]).T


class _DummyRegressor:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load_model(self, _path: str) -> None:
        self.loaded = True

    def predict(self, x):
        import numpy as np

        return np.linspace(0.1, 0.9, len(x), dtype=float)


def _make_draw_df(n: int = 260) -> pd.DataFrame:
    rows = []
    for i in range(n):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {
                "issue": 10000 + i,
                "draw_date": "2026-01-01",
                "numbers": json.dumps(nums, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def test_soft_label_uses_regressor(monkeypatch) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v3_core20"}), encoding="utf-8"
    )
    (MODELS_DIR / "catboost_soft_label.cbm").write_text("x", encoding="utf-8")
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyClassifier)
    monkeypatch.setattr("src.predict.CatBoostRegressor", _DummyRegressor)

    predictor = Predictor.load()
    assert isinstance(predictor.soft_model, _DummyRegressor)


def test_pm1_proximity_uses_classifier(monkeypatch) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v3_core20"}), encoding="utf-8"
    )
    (MODELS_DIR / "catboost_pm1_proximity.cbm").write_text("x", encoding="utf-8")
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyClassifier)
    monkeypatch.setattr("src.predict.CatBoostRegressor", _DummyRegressor)

    predictor = Predictor.load()
    assert isinstance(predictor.pm1_model, _DummyClassifier)


def test_apply_to_top3_only_is_effective() -> None:
    table = pd.DataFrame(
        {
            "number": [28, 27, 29, 44, 55, 10],
            "final_score": [0.95, 0.94, 0.91, 0.85, 0.8, 0.79],
            "local_peak_score": [1.1, 1.08, 1.05, 0.86, 0.8, 0.79],
            "history_prior_score": [0.2, 0.1, 0.15, 0.02, 0.01, 0.01],
            "cand_current_gap_all": [1, 2, 3, 2, 3, 2],
        }
    )
    ranked_top3, summary_top3 = apply_topk_group_dedup(
        table,
        cfg={
            "enabled": True,
            "group_distance": 1,
            "apply_to_top3_only": True,
            "candidate_pool_for_grouping": 20,
        },
        top_k=3,
    )
    ranked_all, summary_all = apply_topk_group_dedup(
        table,
        cfg={
            "enabled": True,
            "group_distance": 1,
            "apply_to_top3_only": False,
            "candidate_pool_for_grouping": 20,
        },
        top_k=3,
    )
    assert (
        ranked_top3["number"].head(5).tolist()
        == table.sort_values("final_score", ascending=False)["number"].head(5).tolist()
    )
    assert (
        ranked_all["number"].head(5).tolist()
        != table.sort_values("final_score", ascending=False)["number"].head(5).tolist()
    )
    assert summary_top3["dedup_applied_scope"] == "top3_only"
    assert summary_all["dedup_applied_scope"] == "ranking"


def test_ablation_calls_real_runtime_logic(monkeypatch) -> None:
    calls = {"local": 0, "dedup": 0}

    def _local(df, cfg, input_score_column, output_score_column):
        calls["local"] += 1
        out = df.copy()
        out["raw_score"] = out[input_score_column]
        out["local_peak_score"] = out[input_score_column]
        out[output_score_column] = out[input_score_column]
        return out, {"enabled": True}

    def _dedup(df, cfg, top_k):
        calls["dedup"] += 1
        top3 = df.sort_values("final_score", ascending=False)["number"].head(3).tolist()
        return df.sort_values("final_score", ascending=False).reset_index(drop=True), {
            "top3_after_group_dedup": top3,
            "top3_before_group_dedup": top3,
            "dedup_applied_scope": "ranking",
        }

    monkeypatch.setattr(backtest, "apply_local_peak_correction", _local)
    monkeypatch.setattr(backtest, "apply_topk_group_dedup", _dedup)

    issues = pd.DataFrame(
        [
            {
                "pred_top3": [28, 44, 55],
                "pred_top10": [28, 44, 55, 11, 12, 13, 14, 15, 16, 17],
                "actual": [27, 45, 54],
                "prev_numbers": [28, 40, 50],
                "history_length": 160,
                "score_table": [
                    {"number": n, "final_score": float(100 - n)} for n in range(1, 81)
                ],
            }
        ]
    )
    _ = _ablation_report_from_issue_rows(issues)
    assert calls["local"] > 0
    assert calls["dedup"] > 0


def test_error_shift_report_separates_exact_and_pm1_error() -> None:
    issues = pd.DataFrame(
        [
            {
                "pred_top3": [27, 34, 50],
                "pred_top10": [27, 34, 50, 10, 11, 12, 13, 14, 15, 16],
                "actual": [27, 33, 51],
                "prev_numbers": [27, 40, 50],
                "history_length": 120,
            }
        ]
    )
    _, summary = _build_error_shift_report(issues)
    assert "zone_pm1_proximity_rate" in summary
    assert "zone_strict_pm1_error_rate" in summary
    prox = {r["zone"]: r["pm1_proximity"] for r in summary["zone_pm1_proximity_rate"]}
    strict = {
        r["zone"]: r["strict_pm1_error"] for r in summary["zone_strict_pm1_error_rate"]
    }
    assert prox
    assert strict


def test_predict_output_contains_full_breakdown(monkeypatch) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v3_core20"}), encoding="utf-8"
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyClassifier)
    monkeypatch.setattr("src.predict.CatBoostRegressor", _DummyRegressor)
    monkeypatch.setattr(
        "src.predict.load_history_snapshot_payload",
        lambda: {
            "status": "unavailable",
            "meta": {},
            "number_priors": pd.DataFrame(),
            "load_elapsed_ms": 1,
        },
    )

    predictor = Predictor.load()
    predictor.runtime_config["history_prior"] = {
        "enabled": True,
        "model_weight": 0.88,
        "history_weight": 0.12,
    }
    predictor.runtime_config["analysis_rerank"] = {
        "enabled": True,
        "top_k": 30,
        "weight": 0.08,
    }
    predictor.runtime_config["topk_group_dedup"] = {
        "enabled": True,
        "group_distance": 1,
        "apply_to_top3_only": True,
        "candidate_pool_for_grouping": 20,
    }

    out = predictor.predict_from_draws(_make_draw_df(260), min_history=22)
    assert "local_peak_summary" in out
    assert "grouped_candidates_preview" in out
    assert "top3_before_group_dedup" in out
    assert "top3_after_group_dedup" in out
    assert "final_score_breakdown" in out
    assert out["raw_score_table"]
    first = out["raw_score_table"][0]
    for key in [
        "model_score",
        "history_prior_score",
        "long_feature_score",
        "soft_label_score",
        "pm1_proximity_score",
        "score_before_analysis_rerank",
        "analysis_compatibility_score",
        "score_after_analysis_rerank",
        "raw_score",
        "local_peak_score",
        "score_after_local_peak",
        "final_score",
        "cand_current_gap_all",
        "rank_model_only",
        "rank_final",
    ]:
        assert key in first
