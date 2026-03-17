from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.predict import Predictor
from src.ranking_dataset import (
    build_ranker_training_rows,
    split_ranker_training_frame,
)
from src.runtime_scoring import (
    RUNTIME_SCORE_REQUIRED_COLUMNS,
    RuntimeScoringOutputs,
    score_candidates_runtime,
)
from src.strategy import StrategyConfig
from src.utils import MODELS_DIR, V3_CORE20_COLUMNS


def _issue_payload(issue: int, target: list[int]) -> dict[str, object]:
    cand = pd.DataFrame(
        {
            "number": np.arange(1, 81, dtype=int),
            "cand_current_gap_all": np.linspace(0.0, 1.0, 80),
            "cand_hits_last_200": np.linspace(0.0, 1.0, 80),
            "cand_hits_last_500": np.linspace(0.0, 1.0, 80),
            "cand_hits_last_1000": np.linspace(0.0, 1.0, 80),
            "cand_total_hits_all_time": np.linspace(0.0, 1.0, 80),
            "cand_avg_gap_all": np.linspace(0.0, 1.0, 80),
            "cand_max_gap_all": np.linspace(0.0, 1.0, 80),
            "cand_today_hits": np.linspace(0.0, 1.0, 80),
            "cand_carryover_from_prev": np.linspace(0.0, 1.0, 80),
            "cand_pm1_neighbor_hits": np.linspace(0.0, 1.0, 80),
            "cand_pm2_neighbor_hits": np.linspace(0.0, 1.0, 80),
            "f1": np.linspace(0.0, 1.0, 80),
            "f2": np.linspace(1.0, 0.0, 80),
        }
    )
    return {
        "cand": cand,
        "target": set(target),
        "issue_row": pd.Series({"issue": issue}),
    }


def test_ranker_dataset_group_blocks_and_columns() -> None:
    payloads = {
        0: _issue_payload(1001, [1, 2, 3]),
        1: _issue_payload(1002, [4, 5, 6]),
    }
    rank_df = build_ranker_training_rows(payloads, [0, 1], ["f1", "f2"])
    assert len(rank_df) == 160
    assert rank_df.columns.tolist() == [
        "issue",
        "number",
        "f1",
        "f2",
        "label",
        "group_id",
    ]
    assert set(rank_df.iloc[:80]["group_id"].unique()) == {1001}
    assert set(rank_df.iloc[80:160]["group_id"].unique()) == {1002}

    x, y, gid = split_ranker_training_frame(rank_df, ["f1", "f2"])
    assert x.columns.tolist() == ["f1", "f2"]
    assert y.dtype.kind == "f"
    assert gid.dtype.kind in {"i", "u"}


def test_runtime_required_columns_present() -> None:
    cand = _issue_payload(1001, [1, 2, 3])["cand"]
    out = score_candidates_runtime(
        base_scores=np.linspace(0.1, 0.9, 80),
        candidate_df=cand,
        recent_draws=[[1, 2, 3]],
        runtime_config={
            "history_prior": {"enabled": True},
            "long_feature_injection": {"enabled": True},
            "analysis_rerank": {"enabled": True, "top_k": 20, "weight": 0.1},
            "neighbor_peak_correction": {"enabled": True},
            "topk_group_dedup": {"enabled": True, "apply_to_top3_only": True},
            "soft_label_training": {"enabled": False},
            "proximity_model": {"enabled": False},
        },
        snapshot_payload={"number_priors": pd.DataFrame(), "meta": {}},
        board_priors={},
        soft_label_raw=None,
        pm1_proximity_raw=None,
    )
    got = out.score_table.columns.tolist()
    for col in RUNTIME_SCORE_REQUIRED_COLUMNS:
        assert col in got


class _DummyClassifier:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load_model(self, _path: str) -> None:
        self.loaded = True

    def predict_proba(self, x):
        vals = np.linspace(0.2, 0.8, len(x), dtype=float)
        return np.vstack([1.0 - vals, vals]).T


class _DummyBrokenRanker:
    def load_model(self, _path: str) -> None:
        raise RuntimeError("broken ranker")


class _DummyRanker(_DummyClassifier):
    def predict(self, x):
        return np.linspace(0.3, 0.9, len(x), dtype=float)


def test_predictor_fallback_when_ranker_missing_or_broken(monkeypatch) -> None:
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
    (MODELS_DIR / "catboost_top20.cbm").write_text("x", encoding="utf-8")
    (MODELS_DIR / "catboost_ranker_top80.cbm").write_text("x", encoding="utf-8")
    (MODELS_DIR / "catboost_soft_ce.cbm").unlink(missing_ok=True)
    (MODELS_DIR / "catboost_pm1_proximity.cbm").unlink(missing_ok=True)
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyClassifier)
    monkeypatch.setattr("src.predict.CatBoostRanker", _DummyBrokenRanker)

    predictor = Predictor.load()
    assert predictor.model is not None
    assert predictor.ranker_model is None
    assert predictor.use_ranker_main is False


def test_cascade_selector_top3_is_protected(monkeypatch) -> None:
    predictor = Predictor(
        model=_DummyClassifier(),
        ranker_model=None,
        soft_model=None,
        pm1_model=None,
        use_ranker_main=False,
        feature_columns=list(V3_CORE20_COLUMNS),
        strategy=StrategyConfig(
            version_id="cascade_v1_flow",
            stage_type="cascade",
            candidate_pool=30,
            prior_window=300,
            rerank_weight=0.0,
            penalty_weight=0.0,
            trend_weight=0.0,
            regime_aware=True,
            pipeline_version="cascade_v1",
            model_artifact_dir="models/cascade_v1",
            stage1_keep=30,
            stage2_keep=10,
        ),
        feature_version="v3_core20",
        runtime_config={
            "topk_group_dedup": {"enabled": True, "apply_to_top3_only": True}
        },
        cascade_pipeline=type(
            "_Cascade",
            (),
            {
                "predict_issue": staticmethod(
                    lambda _row: {
                        "final_scores": np.linspace(0.1, 0.9, 80),
                        "stage1": pd.DataFrame(
                            {
                                "number": np.arange(1, 81),
                                "stage1_score": np.linspace(0.1, 0.9, 80),
                                "stage1_keep_flag": [1] * 30 + [0] * 50,
                            }
                        ),
                        "stage2": pd.DataFrame(
                            {
                                "number": np.arange(1, 81),
                                "stage2_score": np.linspace(0.1, 0.9, 80),
                                "stage2_keep_flag": [1] * 10 + [0] * 70,
                            }
                        ),
                        "stage3_inputs": pd.DataFrame({"number": np.arange(1, 11)}),
                        "final_top3": [7, 27, 47],
                        "no_selector_top3": [1, 2, 3],
                        "selector_score": 1.0,
                        "selector_reason": "test",
                        "selector_regime": "balanced",
                    }
                )
            },
        )(),
    )

    monkeypatch.setattr(
        "src.predict.build_latest_issue_features_for_inference",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "issue": 123,
                    "history_numbers": json.dumps([[1, 2, 3], [4, 5, 6]]),
                    "prev_numbers": json.dumps([1, 2, 3]),
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "src.predict.build_candidate_matrix",
        lambda *_args, **_kwargs: pd.DataFrame(
            {c: np.linspace(0.0, 1.0, 80) for c in V3_CORE20_COLUMNS}
        ),
    )
    monkeypatch.setattr(
        "src.predict.load_history_snapshot_payload",
        lambda: {
            "status": "unavailable",
            "meta": {},
            "number_priors": pd.DataFrame(),
            "load_elapsed_ms": 1,
        },
    )
    monkeypatch.setattr(
        "src.predict.score_candidates_runtime",
        lambda **_kwargs: RuntimeScoringOutputs(
            score_table=pd.DataFrame(
                {
                    "number": np.arange(1, 81),
                    **{
                        c: np.linspace(0.1, 0.9, 80)
                        for c in RUNTIME_SCORE_REQUIRED_COLUMNS
                        if c != "number"
                    },
                    "score": np.linspace(0.1, 0.9, 80),
                }
            ),
            rerank_summary={},
            local_peak_summary={},
            dedup_summary={
                "top3_after_group_dedup": [1, 2, 3],
                "top3_before_group_dedup": [1, 2, 3],
                "grouped_candidates_preview": [],
                "dedup_applied_scope": "top3_only",
            },
        ),
    )

    draws = pd.DataFrame(
        [
            {
                "issue": 1,
                "numbers": json.dumps([1, 2, 3]),
                "draw_date": "2026-01-01",
            },
            {
                "issue": 2,
                "numbers": json.dumps([4, 5, 6]),
                "draw_date": "2026-01-01",
            },
        ]
    )
    out = predictor.predict_from_draws(draws, min_history=1)
    assert out["top3_numbers"] == [7, 27, 47]
    assert out["top3_selector_final"] == [7, 27, 47]
