from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple
import json
import math
from pathlib import Path

Cell = Tuple[int, int]


def normalize_scores_per_module(raw_scores: Mapping[Cell, float], mode: str = "per_module_minmax") -> Dict[Cell, float]:
    if not raw_scores:
        return {}
    if mode != "per_module_minmax":
        raise ValueError(f"Unsupported competitor normalization mode: {mode}")
    values = [float(v) for v in raw_scores.values()]
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return {cell: 0.5 for cell in raw_scores}
    return {cell: (float(v) - lo) / (hi - lo) for cell, v in raw_scores.items()}


def compute_dense_ranks(scores_by_cell: Mapping[Cell, float]) -> Dict[Cell, int]:
    unique = sorted({float(v) for v in scores_by_cell.values()}, reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(unique)}
    return {cell: int(rank_of[float(score)]) for cell, score in scores_by_cell.items()}


def compute_vote_signals(ranks_by_cell: Mapping[Cell, int]) -> Dict[Cell, Dict[str, float]]:
    out: Dict[Cell, Dict[str, float]] = {}
    for cell, rank in ranks_by_cell.items():
        out[cell] = {
            "is_top1": 1.0 if rank <= 1 else 0.0,
            "is_top3": 1.0 if rank <= 3 else 0.0,
            "is_top5": 1.0 if rank <= 5 else 0.0,
        }
    return out


def borda_scores(
    module_ranks: Mapping[str, Mapping[Cell, int]],
    cells: Sequence[Cell],
) -> Dict[Cell, float]:
    if not module_ranks:
        return {cell: 0.0 for cell in cells}
    n = max(len(cells), 1)
    totals = {cell: 0.0 for cell in cells}
    for ranks in module_ranks.values():
        for cell in cells:
            rank = int(ranks.get(cell, n))
            totals[cell] += float(n - rank + 1)
    max_v = max(totals.values()) if totals else 1.0
    min_v = min(totals.values()) if totals else 0.0
    if abs(max_v - min_v) < 1e-12:
        return {cell: 0.5 for cell in cells}
    return {cell: (v - min_v) / (max_v - min_v) for cell, v in totals.items()}


def rrf_scores(
    module_ranks: Mapping[str, Mapping[Cell, int]],
    cells: Sequence[Cell],
    k: float = 10.0,
) -> Dict[Cell, float]:
    totals = {cell: 0.0 for cell in cells}
    if not module_ranks:
        return totals
    n = max(len(cells), 1)
    for ranks in module_ranks.values():
        for cell in cells:
            rank = int(ranks.get(cell, n))
            totals[cell] += 1.0 / (k + rank)
    max_v = max(totals.values()) if totals else 1.0
    min_v = min(totals.values()) if totals else 0.0
    if abs(max_v - min_v) < 1e-12:
        return {cell: 0.5 for cell in cells}
    return {cell: (v - min_v) / (max_v - min_v) for cell, v in totals.items()}


def aggregate_topk_votes(
    module_votes: Mapping[str, Mapping[Cell, Mapping[str, float]]],
    cells: Sequence[Cell],
) -> Dict[Cell, Dict[str, float]]:
    out = {cell: {"top1_vote_count": 0.0, "top3_vote_count": 0.0, "top5_vote_count": 0.0} for cell in cells}
    for votes in module_votes.values():
        for cell in cells:
            row = votes.get(cell, {})
            out[cell]["top1_vote_count"] += float(row.get("is_top1", 0.0))
            out[cell]["top3_vote_count"] += float(row.get("is_top3", 0.0))
            out[cell]["top5_vote_count"] += float(row.get("is_top5", 0.0))
    return out


def build_meta_judge_feature_row(candidate: Mapping[str, Any], feature_names: Sequence[str]) -> Dict[str, float]:
    return {name: float(candidate.get(name, 0.0)) for name in feature_names}


def compute_rank_entropy_like(ranks: Sequence[int]) -> float:
    if not ranks:
        return 0.0
    counts: Dict[int, int] = {}
    for rank in ranks:
        counts[int(rank)] = counts.get(int(rank), 0) + 1
    total = sum(counts.values())
    probs = [c / max(total, 1) for c in counts.values()]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    max_entropy = math.log(max(len(probs), 1))
    return entropy / max(max_entropy, 1e-12)


def load_meta_judge_artifact(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"meta judge artifact not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("meta judge artifact must be an object")
    required = {
        "model_type",
        "feature_names",
        "coef",
        "intercept",
        "schema_version",
        "trained_from_real_data",
        "walk_forward_only",
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"meta judge artifact missing keys: {missing}")
    validate_meta_judge_artifact(data)
    return data


def validate_meta_judge_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("model_type") != "logistic_ranker":
        raise ValueError("meta judge artifact model_type must be logistic_ranker")
    feature_names = artifact.get("feature_names", [])
    coef = artifact.get("coef", [])
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("meta judge artifact feature_names must be non-empty list")
    if not isinstance(coef, list):
        raise ValueError("meta judge artifact coef must be list")
    if len(feature_names) != len(coef):
        raise ValueError("meta judge artifact feature_names and coef size mismatch")
    if not isinstance(artifact.get("intercept"), (int, float)):
        raise ValueError("meta judge artifact intercept must be numeric")
    if not str(artifact.get("schema_version", "")).strip():
        raise ValueError("meta judge artifact schema_version is required")
    if bool(artifact.get("trained_from_real_data")) is not True:
        raise ValueError("meta judge artifact requires trained_from_real_data=true")
    if bool(artifact.get("walk_forward_only")) is not True:
        raise ValueError("meta judge artifact requires walk_forward_only=true")


def score_with_logistic_artifact(feature_row: Mapping[str, float], artifact: Mapping[str, Any]) -> float:
    if artifact.get("model_type") != "logistic_ranker":
        raise ValueError(f"Unsupported judge model type: {artifact.get('model_type')}")
    names = list(artifact.get("feature_names", []))
    coef = [float(x) for x in artifact.get("coef", [])]
    intercept = float(artifact.get("intercept", 0.0))
    if len(names) != len(coef):
        raise ValueError("meta judge artifact feature_names and coef size mismatch")
    z = intercept
    for name, w in zip(names, coef):
        z += float(feature_row.get(name, 0.0)) * w
    return z
