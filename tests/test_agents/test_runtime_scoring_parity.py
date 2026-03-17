from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import src.backtest as backtest
import src.train_lgbm as train_lgbm
from src.predict import Predictor
from src.utils import (
    MODELS_DIR,
    V3_CORE20_COLUMNS,
    build_candidate_matrix,
    build_latest_issue_features_for_inference,
)


class _DummyClassifier:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load_model(self, _path: str) -> None:
        self.loaded = True

    def fit(self, *_args, **_kwargs):
        return None

    def save_model(self, _path: str) -> None:
        return None

    def predict_proba(self, x):
        vals = np.linspace(0.2, 0.8, len(x), dtype=float)
        return np.vstack([1.0 - vals, vals]).T

    def predict(self, x):
        return np.linspace(0.2, 0.8, len(x), dtype=float)

    def get_feature_importance(self):
        return [1.0 for _ in range(len(V3_CORE20_COLUMNS))]


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


def test_backtest_runtime_chain_matches_predict_when_soft_pm1_disabled(
    monkeypatch,
) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "feature_version": "v3_core20",
                "model_type": "catboost_ranker",
                "runtime_config": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyClassifier)
    monkeypatch.setattr("src.predict.CatBoostRanker", _DummyClassifier)
    (MODELS_DIR / "catboost_ranker_top80.cbm").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "src.predict.load_history_snapshot_payload",
        lambda: {
            "status": "unavailable",
            "meta": {},
            "number_priors": pd.DataFrame(),
            "load_elapsed_ms": 1,
        },
    )

    draws = _make_draw_df(260)
    predictor = Predictor.load()
    predictor.runtime_config["soft_label_training"] = {
        "enabled": False,
        "blend_weight": 0.15,
        "normalization": "rank_pct",
    }
    predictor.runtime_config["proximity_model"] = {"enabled": False, "pm1_weight": 0.12}
    out = predictor.predict_from_draws(draws, min_history=22)

    issue_df = build_latest_issue_features_for_inference(draws, min_history=22)
    issue_row = issue_df.iloc[-1]
    cand = build_candidate_matrix(issue_row, V3_CORE20_COLUMNS, strict_features=False)
    cand = cand.reset_index(drop=True)
    cand.insert(0, "number", np.arange(1, 81, dtype=int))
    payload = {"cand": cand, "issue_row": issue_row}
    model_score_map = {
        int(rec["number"]): float(rec["model_score"]) for rec in out["raw_score_table"]
    }
    base_scores = np.array([model_score_map[i] for i in range(1, 81)], dtype=float)
    runtime_bundle = {
        "runtime_config": predictor.runtime_config,
        "snapshot_payload": {
            "status": "unavailable",
            "meta": {},
            "number_priors": pd.DataFrame(),
            "load_elapsed_ms": 1,
        },
        "board_priors": {},
        "soft_model": None,
        "pm1_model": None,
    }
    backtest_out = backtest._score_issue_with_runtime_pipeline(
        payload=payload,
        scores=np.array(base_scores, dtype=float),
        runtime_bundle=runtime_bundle,
    )
    first = backtest_out.score_table.iloc[0].to_dict()
    for key in [
        "model_score",
        "history_prior_score",
        "long_feature_score",
        "soft_label_score",
        "pm1_proximity_score",
        "score_before_analysis_rerank",
        "analysis_compatibility_score",
        "analysis_rerank_score",
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
    top10_bt = backtest_out.score_table["number"].head(10).astype(int).tolist()
    top3_bt = backtest_out.dedup_summary["top3_after_group_dedup"]

    top10_predict_runtime = [int(x["number"]) for x in out["score_table"][:10]]
    assert top10_predict_runtime == top10_bt
    assert out["top3_after_group_dedup"] == top3_bt


def test_training_metadata_consistency_for_soft_pm1(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        train_lgbm,
        "_parse_args",
        lambda: SimpleNamespace(
            debug=True,
            max_issues=None,
            iterations=None,
            watchdog_seconds=0,
            thread_count=None,
        ),
    )
    monkeypatch.setattr(
        train_lgbm,
        "build_training_frame",
        lambda cfg, args, monitor: (
            pd.DataFrame(
                [
                    {
                        "issue": 1,
                        "target_issue": 2,
                        "target_numbers": json.dumps([1, 2, 3]),
                        "history_numbers": json.dumps([[1, 2, 3]]),
                        "current_numbers": json.dumps([1, 2, 3]),
                        "prev_numbers": json.dumps([1, 2, 3]),
                    }
                ]
            ),
            list(V3_CORE20_COLUMNS),
            "v3_core20",
        ),
    )
    monkeypatch.setattr(train_lgbm, "_load_experiments", lambda: [])
    monkeypatch.setattr(
        train_lgbm,
        "_evaluate_strategies",
        lambda *args, **kwargs: (
            pd.DataFrame(
                [
                    {
                        "version_id": "v0_binary_baseline",
                        "keep_recommendation": True,
                        "top3_at_least_one_hit_rate": 0.1,
                        "top3_hit_rate": 0.1,
                        "top20_hit_rate": 0.1,
                        "top10_hit_rate": 0.1,
                        "top5_hit_rate": 0.1,
                        "train_vs_oos_gap": 0.0,
                        "fold_dispersion": 0.0,
                        "regime_dispersion": 0.0,
                        "is_overfit": False,
                    }
                ]
            ),
            {"version_id": "v0_binary_baseline"},
            {"version_id": "v0_binary_baseline"},
        ),
    )
    monkeypatch.setattr(
        train_lgbm, "_select_formal_strategy", lambda df: df.iloc[0].to_dict()
    )
    monkeypatch.setattr(
        train_lgbm,
        "precompute_issue_payloads",
        lambda *args, **kwargs: {
            0: {
                "cand": pd.DataFrame({c: [0.0] * 80 for c in V3_CORE20_COLUMNS}),
                "target": {1, 2, 3},
                "regime": "balanced",
                "issue_row": pd.Series({"history_numbers": "[]"}),
            }
        },
    )
    monkeypatch.setattr(
        train_lgbm,
        "_expand_rows",
        lambda payloads, idx: (
            pd.DataFrame({c: [0.0] * 80 for c in V3_CORE20_COLUMNS}),
            pd.Series([0] * 80),
        ),
    )
    monkeypatch.setattr(train_lgbm, "CatBoostClassifier", _DummyClassifier)
    monkeypatch.setattr(train_lgbm, "CatBoostRanker", _DummyClassifier)

    def _fake_yaml(_p):
        return {
            "pipeline": {"version": "baseline_flat_score"},
            "training_mode": "ranker_main",
            "ranking_experiment": {
                "enabled": True,
                "objective": "QuerySoftMax",
                "eval_metric": "NDCG:top=10",
                "custom_metrics": ["NDCG:top=3"],
            },
            "catboost_params": {"iterations": 1, "verbose": False},
            "soft_label_training": {
                "enabled": True,
                "pm1_weight": 0.35,
                "pm2_weight": 0.15,
                "blend_weight": 0.15,
                "normalization": "rank_pct",
            },
            "proximity_model": {"enabled": True, "pm1_weight": 0.12},
            "research_backtest_splits": 2,
            "backtest_splits": 2,
            "research_iterations": 1,
            "final_stage_versions": 1,
            "overfit_thresholds": {},
        }

    monkeypatch.setattr(train_lgbm, "load_yaml", _fake_yaml)

    def _capture_save_json(path, payload):
        if str(path).endswith("models/metadata.json"):
            captured["metadata"] = payload

    monkeypatch.setattr(train_lgbm, "save_json", _capture_save_json)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *args, **kwargs: None)

    train_lgbm.main()
    meta = captured["metadata"]
    for key in [
        "soft_label_model_path",
        "pm1_model_path",
        "soft_label_training",
        "proximity_model",
        "soft_label_normalization_method",
        "train_rows_used",
        "ranking_objective",
        "ranking_eval_metric",
        "group_count",
    ]:
        assert key in meta
    assert "ranker" in meta["train_rows_used"]
