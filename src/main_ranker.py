from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

from src.ranking_features import build_candidate_feature_rows


class MainRankerError(ValueError):
    pass


@dataclass
class ResolvedModel:
    size_class: str
    model_strategy: str
    model_used: str
    fallback_used: bool
    fallback_reason: Optional[str]
    backend: str
    artifact_path: Path
    feature_columns: List[str]


REGISTRY_PATH = Path("artifacts/model_registry.json")


def _normalize_artifact_path(raw: str) -> Path:
    return Path(str(raw).replace("\\", "/"))


def _size_class_of_board(board: List[List[int]]) -> str:
    return f"{len(board)}x{len(board[0])}"


def load_model_registry(path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise MainRankerError(f"model registry missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_for_size(
    board: List[List[int]],
    strict_missing_artifact: bool = True,
    path: Path = REGISTRY_PATH,
) -> ResolvedModel:
    size_class = _size_class_of_board(board)
    reg = load_model_registry(path)
    strategy = reg.get("model_strategy", "size_specific_with_global_fallback")
    per_size = reg.get("per_size", {})
    global_cfg = reg.get("global", {})

    size_item = per_size.get(size_class)
    if size_item:
        artifact = _normalize_artifact_path(size_item["artifact_path"])
        if artifact.exists():
            return ResolvedModel(
                size_class=size_class,
                model_strategy=strategy,
                model_used=f"size:{size_class}",
                fallback_used=False,
                fallback_reason=None,
                backend=size_item.get("backend", "unknown"),
                artifact_path=artifact,
                feature_columns=list(size_item.get("feature_columns", [])),
            )

    fallback_reason = "size_specific_model_missing"
    g_artifact = _normalize_artifact_path(global_cfg.get("artifact_path", "")) if global_cfg else Path("")
    if global_cfg and g_artifact.exists():
        return ResolvedModel(
            size_class=size_class,
            model_strategy=strategy,
            model_used="global",
            fallback_used=True,
            fallback_reason=fallback_reason,
            backend=global_cfg.get("backend", "unknown"),
            artifact_path=g_artifact,
            feature_columns=list(global_cfg.get("feature_columns", [])),
        )

    if strict_missing_artifact:
        raise MainRankerError(
            "missing artifacts for "
            f"size={size_class}; size_model={bool(size_item)} global={bool(global_cfg)}"
        )

    raise MainRankerError("no trained model available")


def score_candidates_with_ranker(
    board: List[List[int]],
    target_number: int,
    candidates: List[Tuple[int, int]],
    *,
    strict_missing_artifact: bool = True,
    registry_path: Path = REGISTRY_PATH,
) -> Tuple[List[float], Dict[str, Any]]:
    resolved = resolve_model_for_size(
        board,
        strict_missing_artifact=strict_missing_artifact,
        path=registry_path,
    )
    artifact = joblib.load(resolved.artifact_path)
    model = artifact["model"]
    feature_columns = resolved.feature_columns or artifact.get("feature_columns", [])
    if not feature_columns:
        raise MainRankerError("feature columns missing")

    candidate_rows = [
        {
            "row": r + 1,
            "col": c + 1,
            "score": 0.0,
            "module_scores": {},
            "module_details": {},
            "module_informative": {},
        }
        for r, c in candidates
    ]
    feat_rows = build_candidate_feature_rows(
        case_id=f"runtime:{target_number}",
        board_shape=(len(board), len(board[0])),
        candidates=candidate_rows,
        true_cell_1_based=None,
        board=board,
        target_number=target_number,
    )
    x = [[float(row.get(col, 0.0)) for col in feature_columns] for row in feat_rows]

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1].tolist()
    else:
        scores = model.predict(x).tolist()

    meta = {
        "model_strategy": resolved.model_strategy,
        "model_used": resolved.model_used,
        "size_class": resolved.size_class,
        "fallback_used": resolved.fallback_used,
        "fallback_reason": resolved.fallback_reason,
        "backend": resolved.backend,
        "registry_path": str(registry_path),
        "artifact_path": str(resolved.artifact_path),
    }
    return [float(s) for s in scores], meta


def write_model_registry(registry: Dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = dict(registry)
    registry.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
