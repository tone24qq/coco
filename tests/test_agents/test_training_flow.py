import json

import pandas as pd
import pytest

import src.build_features as build_features
from src.predict import _load_strategy_payload


def test_load_strategy_payload_prefers_strategy_config(tmp_path, monkeypatch) -> None:
    strategy_payload = {"selected_strategy": {"version_id": "v4_two_stage_20_10_3"}}
    metadata_payload = {"selected_strategy": {"version_id": "v0_binary_baseline"}}
    (tmp_path / "strategy_config.json").write_text(
        json.dumps(strategy_payload), encoding="utf-8"
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata_payload), encoding="utf-8"
    )
    monkeypatch.setattr("src.predict.MODELS_DIR", tmp_path)

    out = _load_strategy_payload()

    assert out["selected_strategy"]["version_id"] == "v4_two_stage_20_10_3"


def test_build_features_rejects_too_small_max_draws(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(build_features, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(build_features, "FEATURE_STORE_DIR", tmp_path)
    monkeypatch.setattr(build_features, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        build_features,
        "load_yaml",
        lambda _: {"max_draws_for_training": 2999, "feature_min_history": 22},
    )
    dummy_draws = pd.DataFrame(
        {
            "issue": [1, 2],
            "draw_date": ["2026-01-01", "2026-01-01"],
            "numbers": ["[]", "[]"],
        }
    )
    monkeypatch.setattr(build_features, "load_processed", lambda: dummy_draws)

    with pytest.raises(ValueError, match="不可低於 3000"):
        build_features.main()
