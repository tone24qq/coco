import json

import pandas as pd
import pytest

from src.build_features import main as build_features_main
from src.predict import Predictor
from src.utils import MODELS_DIR, precompute_issue_payloads


def test_precompute_issue_payloads_builds_cached_candidate_and_target() -> None:
    feature_df = pd.DataFrame(
        [
            {
                "issue": 1,
                "target_numbers": json.dumps([1, 2, 3]),
                "history_numbers": json.dumps([[1, 2, 3]]),
                "current_numbers": json.dumps([1, 2, 3]),
                "prev_numbers": json.dumps([1, 2, 3]),
                "freq_last_10": 0.0,
                "ema_short_minus_ema_long": 0.0,
                "cooccur_with_last_draw_mean": 0.0,
                "num_zone": 0.0,
            }
        ]
    )
    feature_columns = [
        "freq_last_10",
        "ema_short_minus_ema_long",
        "cooccur_with_last_draw_mean",
        "num_zone",
        "num",
    ]
    payloads = precompute_issue_payloads(feature_df, feature_columns)

    assert 0 in payloads
    assert payloads[0]["cand"].shape[0] == 80
    assert payloads[0]["target"] == {1, 2, 3}


def test_precompute_issue_payloads_strict_raises_for_missing_feature() -> None:
    feature_df = pd.DataFrame(
        [
            {
                "issue": 1,
                "target_numbers": json.dumps([1, 2, 3]),
                "history_numbers": json.dumps([[1, 2, 3]]),
                "current_numbers": json.dumps([1, 2, 3]),
                "prev_numbers": json.dumps([1, 2, 3]),
            }
        ]
    )
    with pytest.raises(ValueError):
        precompute_issue_payloads(
            feature_df,
            ["missing_col"],
            strict_features=True,
        )


def test_build_features_raises_when_max_draws_less_than_3000(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.build_features.load_yaml",
        lambda _p: {"max_draws_for_training": 2999, "feature_min_history": 22},
    )
    monkeypatch.setattr(
        "src.build_features.load_processed",
        lambda: pd.DataFrame({"issue": [1], "numbers": [json.dumps([1] * 20)]}),
    )

    with pytest.raises(ValueError, match="max_draws_for_training"):
        build_features_main()


def test_predictor_prefers_strategy_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg = {
        "selected_strategy": {
            "version_id": "v4_two_stage_20_10_3",
            "stage_type": "two_stage",
            "candidate_pool": 20,
            "prior_window": 300,
            "rerank_weight": 3.0,
            "penalty_weight": 0.11,
            "trend_weight": 0.4,
            "regime_aware": True,
        }
    }
    (MODELS_DIR / "strategy_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"selected_strategy": {"version_id": "v0_binary_baseline"}}),
        encoding="utf-8",
    )
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(["num"]), encoding="utf-8"
    )

    class DummyModel:
        def load_model(self, _path: str) -> None:
            return None

    monkeypatch.setattr("src.predict.CatBoostClassifier", DummyModel)
    predictor = Predictor.load()

    assert predictor.strategy.version_id == "v4_two_stage_20_10_3"
