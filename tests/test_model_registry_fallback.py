from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from src.main_ranker import MainRankerError, resolve_model_for_size


def _dummy_model(path: Path) -> None:
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]
    m = HistGradientBoostingClassifier(max_iter=5)
    m.fit(x, y)
    joblib.dump({"model": m, "feature_columns": ["board_state_a"], "backend": "sklearn"}, path)


def test_fallback_to_global(tmp_path: Path) -> None:
    g = tmp_path / "global.pkl"
    _dummy_model(g)
    reg = {
        "model_strategy": "size_specific_with_global_fallback",
        "global": {
            "artifact_path": str(g),
            "backend": "sklearn",
            "feature_columns": ["board_state_a"],
        },
        "per_size": {
            "8x10": {
                "artifact_path": str(tmp_path / "missing.pkl"),
                "backend": "sklearn",
                "feature_columns": ["board_state_a"],
            }
        },
    }
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg), encoding="utf-8")
    res = resolve_model_for_size([[1, 2], [3, 4]], path=p)
    assert res.model_used == "global"
    assert res.fallback_used is True


def test_strict_fail_when_missing(tmp_path: Path) -> None:
    reg = {"global": {"artifact_path": str(tmp_path / 'none.pkl')}, "per_size": {}}
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg), encoding="utf-8")
    try:
        resolve_model_for_size([[1]], path=p, strict_missing_artifact=True)
        assert False
    except MainRankerError:
        assert True
