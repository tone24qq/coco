from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from src.utils import DataContractError


@dataclass(frozen=True)
class ModelArtifacts:
    ranker: lgb.Booster
    logistic: Any
    feature_columns: list[str]
    metadata: dict[str, Any]


def load_artifacts(models_dir: Path = Path("models")) -> ModelArtifacts:
    ranker_path = models_dir / "lightgbm_ranker.txt"
    logistic_path = models_dir / "logistic_regression.pkl"
    features_path = models_dir / "feature_columns.json"
    metadata_path = models_dir / "metadata.json"

    required = [ranker_path, logistic_path, features_path, metadata_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise DataContractError(f"missing required artifacts: {missing}")

    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
    if not isinstance(feature_columns, list) or not feature_columns:
        raise DataContractError("feature_columns.json must be a non-empty list")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ranker = lgb.Booster(model_file=str(ranker_path))
    logistic = joblib.load(logistic_path)
    return ModelArtifacts(ranker=ranker, logistic=logistic, feature_columns=feature_columns, metadata=metadata)
