from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.predict import Predictor
from src.utils import (
    MODELS_DIR,
    V3_CORE20_COLUMNS,
    build_candidate_matrix,
    build_issue_features,
)


class _DummyModel:
    def load_model(self, _path: str) -> None:
        return None

    def predict_proba(self, x):
        # deterministic increasing score by number index proxy via row index
        vals = np.linspace(0.1, 0.9, len(x), dtype=float)
        return np.vstack([1.0 - vals, vals]).T


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


def test_candidate_matrix_contains_long_history_columns() -> None:
    df = _make_draw_df(260)
    feat_df = build_issue_features(df, min_history=22)
    x = build_candidate_matrix(feat_df.iloc[-1], V3_CORE20_COLUMNS)
    for col in [
        "cand_hits_last_200",
        "cand_hits_last_500",
        "cand_hits_last_1000",
        "cand_total_hits_all_time",
        "cand_current_gap_all",
        "cand_avg_gap_all",
        "cand_max_gap_all",
        "cand_today_hits",
        "cand_carryover_from_prev",
        "cand_pm1_neighbor_hits",
        "cand_pm2_neighbor_hits",
    ]:
        assert col in x.columns


def test_predictor_outputs_history_prior_and_rerank_summary(monkeypatch) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v3_core20"}), encoding="utf-8"
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyModel)
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
    predictor.runtime_config["long_feature_injection"] = {
        "enabled": True,
        "weight": 0.06,
    }

    out = predictor.predict_from_draws(_make_draw_df(260), min_history=22)
    assert "history_prior_score_summary" in out
    assert "analysis_rerank_summary" in out
    assert "top20_numbers_model" in out
    assert len(out["top20_numbers"]) == 20
