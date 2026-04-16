from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from src.inference_config import load_trained_ranker_config
from src.ranking_features import FEATURE_SCHEMA_VERSION, feature_columns_from_rows

ARTIFACTS_DIR = Path("artifacts")
WEIGHTS_PATH = ARTIFACTS_DIR / "reranker_weights.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "reranker_feature_columns.json"
MODEL_PATH = ARTIFACTS_DIR / "reranker_model.txt"
MAIN_RANKER_PATH = ARTIFACTS_DIR / "main_ranker.pkl"


DEFAULT_FALLBACK_REASON = "reranker_artifact_missing"


def load_reranker_artifact() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not WEIGHTS_PATH.exists():
        return None, DEFAULT_FALLBACK_REASON
    data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    if not data.get("enabled", False):
        return data, data.get("fallback_reason", "reranker_disabled")
    if data.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        return None, "reranker_feature_schema_mismatch"
    return data, None


def apply_reranker(
    candidates: List[Dict[str, Any]],
    feature_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trained_cfg = load_trained_ranker_config()
    if trained_cfg.get("enabled", True):
        if not MAIN_RANKER_PATH.exists():
            if bool(trained_cfg.get("strict_missing_artifact", True)):
                raise ValueError("trained ranker artifact missing: artifacts/main_ranker.pkl")
            return candidates, {
                "ranking_stage": "baseline_only",
                "reranker_version": None,
                "reranker_feature_schema_version": FEATURE_SCHEMA_VERSION,
                "reranker_fallback_reason": "trained_ranker_artifact_missing",
            }
        artifact = joblib.load(MAIN_RANKER_PATH)
        model = artifact["model"]
        feature_columns = artifact.get("feature_columns", [])
        if not feature_columns:
            raise ValueError("trained ranker artifact has empty feature_columns")
        matrix = [[float(row.get(col, 0.0)) for col in feature_columns] for row in feature_rows]
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(matrix)[:, 1]
        else:
            scores = model.predict(matrix)
        rescored = []
        for cand, score in zip(candidates, scores):
            merged = dict(cand)
            merged["reranker_score"] = float(score)
            rescored.append(merged)
        rescored.sort(key=lambda x: x["reranker_score"], reverse=True)
        return rescored, {
            "ranking_stage": "trained_ranker_applied",
            "reranker_version": artifact.get("backend", "main_ranker"),
            "reranker_feature_schema_version": FEATURE_SCHEMA_VERSION,
            "reranker_fallback_reason": None,
        }

    artifact, reason = load_reranker_artifact()
    if artifact is None:
        return candidates, {
            "ranking_stage": "baseline_only",
            "reranker_version": None,
            "reranker_feature_schema_version": FEATURE_SCHEMA_VERSION,
            "reranker_fallback_reason": reason,
        }
    if not artifact.get("enabled", False):
        return candidates, {
            "ranking_stage": "baseline_only",
            "reranker_version": artifact.get("version"),
            "reranker_feature_schema_version": artifact.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            "reranker_fallback_reason": reason,
        }

    weights: Dict[str, float] = artifact.get("weights", {})
    model_type = artifact.get("model_type", "weight_search")
    schema_cols = set(artifact.get("feature_columns", []))
    row_cols = set(feature_columns_from_rows(feature_rows))
    if schema_cols and not schema_cols.issubset(row_cols):
        return candidates, {
            "ranking_stage": "baseline_only",
            "reranker_version": artifact.get("version"),
            "reranker_feature_schema_version": artifact.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            "reranker_fallback_reason": "reranker_feature_columns_missing",
        }

    rescored = []
    for cand, feat in zip(candidates, feature_rows):
        score = 0.0
        for col, w in weights.items():
            score += float(feat.get(col, 0.0)) * float(w)
        merged = dict(cand)
        merged["reranker_score"] = score
        rescored.append(merged)

    rescored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return rescored, {
        "ranking_stage": "reranker_applied",
        "reranker_version": artifact.get("version", model_type),
        "reranker_feature_schema_version": artifact.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        "reranker_fallback_reason": None,
    }
