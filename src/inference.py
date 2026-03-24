"""Runtime artifact loader and deterministic predictor."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.runtime_history import ARTIFACT_VERSION, METADATA_FILENAME

DEFAULT_RUNTIME_DIR = Path(os.getenv("COCO_RUNTIME_DIR", "data/runtime_history"))


def _load_metadata(runtime_dir: Path) -> Dict[str, object]:
    metadata_path = runtime_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)

    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            "Artifact version mismatch: "
            f"expected {ARTIFACT_VERSION}, got {metadata.get('artifact_version')}"
        )

    if metadata.get("score_chain_size") != 80:
        raise ValueError(
            "Artifact schema mismatch: score_chain_size must be 80, "
            f"got {metadata.get('score_chain_size')}"
        )

    return metadata


def _load_scores(
    runtime_dir: Path, metadata: Dict[str, object]
) -> List[Dict[str, float]]:
    score_artifact = metadata.get("score_artifact")
    if not isinstance(score_artifact, str) or not score_artifact:
        raise ValueError("Artifact schema mismatch: invalid score_artifact")

    score_path = runtime_dir / score_artifact
    if not score_path.exists():
        raise FileNotFoundError(f"Missing score artifact: {score_path}")

    df = pd.read_csv(score_path)
    required = {"number", "score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Score artifact schema mismatch, missing: {missing}")

    df["number"] = pd.to_numeric(df["number"], errors="raise").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)
    df = df.sort_values(["number"], kind="mergesort").reset_index(drop=True)

    if df["number"].tolist() != list(range(1, 81)):
        raise ValueError("Score chain is not complete for numbers 1..80")

    return [
        {"number": int(row.number), "score": float(row.score)}
        for row in df.itertuples(index=False)
    ]


def _rank_top20(scores: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    return sorted(scores, key=lambda item: (-item["score"], item["number"]))[:20]


def _combo_metrics(
    combo: Tuple[Dict[str, float], ...], ranks: Dict[int, int]
) -> Tuple[int, int, int, int]:
    numbers = [int(item["number"]) for item in combo]
    tail_unique = len({num % 10 for num in numbers})
    has_low = any(num <= 40 for num in numbers)
    has_high = any(num >= 41 for num in numbers)
    cross_zone = 1 if has_low and has_high else 0

    sorted_numbers = sorted(numbers)
    adjacency_pairs = sum(
        1
        for left, right in zip(sorted_numbers, sorted_numbers[1:])
        if right - left == 1
    )

    rank_sum = sum(ranks[num] for num in numbers)
    return (tail_unique, cross_zone, -adjacency_pairs, -rank_sum)


def _select_top3(top20: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    if len(top20) < 3:
        raise ValueError(
            "Insufficient candidates: top20 must contain at least 3 entries"
        )

    ranks = {int(item["number"]): idx for idx, item in enumerate(top20)}
    best_combo = max(
        itertools.combinations(top20, 3),
        key=lambda combo: _combo_metrics(combo, ranks),
    )
    return sorted(best_combo, key=lambda item: ranks[int(item["number"])])


def predict(runtime_dir: Path | None = None) -> Dict[str, object]:
    resolved_dir = runtime_dir or DEFAULT_RUNTIME_DIR
    metadata = _load_metadata(resolved_dir)
    scores = _load_scores(resolved_dir, metadata)
    top20 = _rank_top20(scores)
    top3 = _select_top3(top20)

    return {
        "latest_issue": metadata.get("latest_issue"),
        "scores": scores,
        "top20": top20,
        "top3": top3,
    }
