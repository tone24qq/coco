from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb

from src.utils import DataContractError


def _load_json_payload(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.loads(fh.read())
    dataset_dir = path.with_suffix(".dataset")
    manifest = dataset_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("format") != "json":
            raise DataContractError(f"unsupported json manifest format: {payload.get('format')}")
        shards = payload.get("shards") or []
        out: list[Any] = []
        for name in shards:
            shard_path = dataset_dir / str(name)
            if not shard_path.exists():
                raise DataContractError(f"json shard missing: {shard_path}")
            with gzip.open(shard_path, "rt", encoding="utf-8") as fh:
                part = json.loads(fh.read())
            if isinstance(part, list):
                out.extend(part)
            else:
                out.append(part)
        return out
    raise DataContractError(f"artifact json not found: {path}")


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

    required = [ranker_path, logistic_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise DataContractError(f"missing required artifacts: {missing}")

    feature_columns = _load_json_payload(features_path)
    if not isinstance(feature_columns, list) or not feature_columns:
        raise DataContractError("feature_columns.json must be a non-empty list")

    metadata = _load_json_payload(metadata_path)
    ranker = lgb.Booster(model_file=str(ranker_path))
    logistic = joblib.load(logistic_path)
    return ModelArtifacts(ranker=ranker, logistic=logistic, feature_columns=feature_columns, metadata=metadata)
