import json
import logging

import numpy as np
import pandas as pd

from src.predict import Predictor
from src.utils import MODELS_DIR, V3_CORE20_COLUMNS, build_issue_features


class _DummyModel:
    def load_model(self, _path: str) -> None:
        return None

    def predict_proba(self, x):
        return np.array([[0.1, 0.9] for _ in range(len(x))], dtype=float)


def _make_draw_df(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        nums = [((i + k) % 80) + 1 for k in range(20)]
        rows.append(
            {"issue": 9000 + i, "draw_date": "2026-01-01", "numbers": json.dumps(nums)}
        )
    return pd.DataFrame(rows)


def test_predictor_uses_metadata_feature_version_for_v3(monkeypatch) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "feature_version": "v3_core20",
                "runtime_config": {
                    "feature_version": "v3_core20",
                    "core_windows": {
                        "z_window": 50,
                        "freq_short": 20,
                        "freq_long": 200,
                        "pmi_window": 200,
                        "handoff_window": 200,
                    },
                    "smoothing_alpha": 0.5,
                    "decay_half_lives": {"ewma": 50, "recent_hit": 5, "neighbor": 10},
                    "distance_kernel_tau": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyModel)
    p = Predictor.load()
    out = p.predict_from_draws(_make_draw_df(), min_history=22)
    assert len(out["top20_numbers"]) == 20


def test_predictor_warns_on_metadata_yaml_mismatch(monkeypatch, caplog) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v2_legacy"}), encoding="utf-8"
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyModel)
    caplog.set_level(logging.WARNING)
    Predictor.load()
    assert "metadata=" in caplog.text and "using metadata" in caplog.text


def test_predict_non_strict_missing_feature_warns_not_crash(
    monkeypatch, caplog
) -> None:
    df = _make_draw_df()
    build_issue_features(df, min_history=22)

    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS + ["prev_numbers"]), encoding="utf-8"
    )
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v2_legacy"}), encoding="utf-8"
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyModel)
    p = Predictor.load()

    caplog.set_level(logging.WARNING)
    out = p.predict_from_draws(df, min_history=22)
    assert len(out["top20_numbers"]) == 20
    assert "Missing feature columns" in caplog.text
