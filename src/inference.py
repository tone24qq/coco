"""Runtime refresh inference pipeline with transformer ranking."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml  # type: ignore[import-untyped]

from src.build_rank_windows import build_inference_window
from src.fetch_latest import FetchConfig, fetch_latest
from src.history_store import load_local_history, merge_history
from src.model_transformer import SmallTransformerRanker, TransformerConfig
from src.normalize_latest import normalize_latest_records
from src.runtime_history import ARTIFACT_VERSION, METADATA_FILENAME

CONFIG_PATH = Path("configs/predict.yaml")


def _load_predict_config(config_path: Path | None = None) -> Dict[str, Any]:
    resolved_path = config_path or CONFIG_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing predict config: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if not isinstance(cfg, dict):
        raise ValueError("Predict config schema mismatch")
    return cfg


def _load_runtime_metadata(runtime_dir: Path) -> Dict[str, Any]:
    metadata_path = runtime_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            "Artifact version mismatch: expected "
            f"{ARTIFACT_VERSION}, got {metadata.get('artifact_version')}"
        )
    return metadata


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
        raise ValueError("Insufficient candidates for top3")
    ranks = {int(item["number"]): idx for idx, item in enumerate(top20)}
    best_combo = max(
        itertools.combinations(top20, 3),
        key=lambda combo: _combo_metrics(combo, ranks),
    )
    return sorted(best_combo, key=lambda item: ranks[int(item["number"])])


def predict() -> Dict[str, object]:
    cfg = _load_predict_config()

    fetch_cfg = cfg.get("fetch")
    runtime_cfg = cfg.get("runtime")
    model_cfg = cfg.get("model")
    sources = cfg.get("auto_fetch_sources")

    if not isinstance(fetch_cfg, dict):
        raise ValueError("Predict config missing fetch section")
    if not isinstance(runtime_cfg, dict):
        raise ValueError("Predict config missing runtime section")
    if not isinstance(model_cfg, dict):
        raise ValueError("Predict config missing model section")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Predict config missing auto_fetch_sources")

    local_history_path = Path(str(runtime_cfg.get("local_history_path", "")))
    runtime_dir = Path(str(runtime_cfg.get("runtime_dir", "")))
    if not str(local_history_path):
        raise ValueError("runtime.local_history_path is required")
    if not str(runtime_dir):
        raise ValueError("runtime.runtime_dir is required")

    latest_records, data_source, fetch_attempts = fetch_latest(
        sources=sources,
        config=FetchConfig(
            timeout_seconds=float(fetch_cfg.get("timeout_seconds", 8.0)),
            retries=int(fetch_cfg.get("retries", 2)),
            backoff_seconds=float(fetch_cfg.get("backoff_seconds", 0.5)),
        ),
    )

    latest_df = normalize_latest_records(latest_records)
    local_df = load_local_history(local_history_path)
    merged_df = merge_history(local_df, latest_df)

    model_file = str(model_cfg.get("artifact_file", ""))
    model_version = str(model_cfg.get("model_version", ""))
    feature_version = str(model_cfg.get("feature_version", ""))
    window_size = int(model_cfg.get("window_size", 100))
    seed = int(model_cfg.get("seed", 42))

    if not model_file or not model_version or not feature_version:
        raise ValueError(
            "model config missing artifact_file/model_version/feature_version"
        )

    metadata = _load_runtime_metadata(runtime_dir)
    if metadata.get("model_version") != model_version:
        raise ValueError(
            "Model version mismatch: expected "
            f"{model_version}, got {metadata.get('model_version')}"
        )
    if metadata.get("feature_version") != feature_version:
        raise ValueError(
            "Feature version mismatch: expected "
            f"{feature_version}, got {metadata.get('feature_version')}"
        )

    model_path = runtime_dir / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    window = build_inference_window(merged_df, window_size=window_size)
    model = SmallTransformerRanker.load(TransformerConfig(seed=seed), str(model_path))
    raw_scores = model.predict_scores(window.features)

    scores = [
        {"number": int(number), "score": float(score)}
        for number, score in zip(window.number_ids, raw_scores)
    ]
    top20 = _rank_top20(scores)
    top3 = _select_top3(top20)

    return {
        "latest_known_issue": window.issue,
        "target_issue": window.target_issue,
        "model_version": model_version,
        "feature_version": feature_version,
        "data_source": data_source,
        "fetch_attempts": fetch_attempts,
        "scores": scores,
        "top20": top20,
        "top3": top3,
        "score_semantics": "ranking_score",
    }
