from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.predict import Predictor
from src.ranking_dataset import build_ranker_training_rows, split_ranker_training_frame
from src.runtime_scoring import RUNTIME_SCORE_REQUIRED_COLUMNS, score_candidates_runtime
from src.utils import MODELS_DIR, V3_CORE20_COLUMNS


class _DummyRanker:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load_model(self, _path: str) -> None:
        self.loaded = True

    def predict(self, x):
        return np.linspace(0.3, 0.9, len(x), dtype=float)


class _DummyClassifier:
    def __init__(self, *args, **kwargs):
        self.loaded = False

    def load_model(self, _path: str) -> None:
        self.loaded = True

    def predict_proba(self, x):
        vals = np.linspace(0.2, 0.8, len(x), dtype=float)
        return np.vstack([1.0 - vals, vals]).T


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
    frame = build_ranker_training_rows(payloads, [0, 1], ["f1", "f2"])
    assert len(frame) == 160
    assert frame.columns.tolist() == [
        "issue",
        "number",
        "f1",
        "f2",
        "label",
        "group_id",
    ]
    assert frame.iloc[:80]["group_id"].nunique() == 1
    assert frame.iloc[80:160]["group_id"].nunique() == 1

    x, y, gid = split_ranker_training_frame(frame, ["f1", "f2"])
    assert x.columns.tolist() == ["f1", "f2"]
    assert y.dtype == float
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
    for col in RUNTIME_SCORE_REQUIRED_COLUMNS:
        assert col in out.score_table.columns


def test_predictor_fail_fast_when_ranker_artifact_missing(monkeypatch) -> None:
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
    (MODELS_DIR / "catboost_ranker_top80.cbm").unlink(missing_ok=True)
    monkeypatch.setattr("src.predict.CatBoostRanker", _DummyRanker)

    with pytest.raises(ValueError, match="ranker artifact missing"):
        Predictor.load()


def test_predictor_uses_ranker_only(monkeypatch) -> None:
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
    (MODELS_DIR / "catboost_ranker_top80.cbm").write_text("x", encoding="utf-8")
    monkeypatch.setattr("src.predict.CatBoostRanker", _DummyRanker)
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyClassifier)

    predictor = Predictor.load()
    assert predictor.use_ranker_main is True
    assert predictor.ranker_model is not None
    assert predictor.model is None
