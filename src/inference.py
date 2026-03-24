"""Runtime inference pipeline using trained PyTorch transformer checkpoint."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]

from src.build_rank_windows import (
    FEATURE_VERSION,
    TENSOR_CONTRACT,
    build_inference_window,
)
from src.fetch_latest import FetchConfig, fetch_latest
from src.history_store import load_local_history, merge_history
from src.model_transformer import SmallTransformerRanker, TransformerConfig
from src.normalize_latest import normalize_latest_records
from src.runtime_history import METADATA_FILENAME

CONFIG_PATH = Path("configs/predict.yaml")


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _load_predict_config(path: Path | None = None) -> Dict[str, Any]:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Missing predict config: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Predict config schema mismatch")
    return cfg


def _load_runtime_metadata(runtime_dir: Path) -> Dict[str, Any]:
    meta_path = runtime_dir / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_transformer_metadata(
    runtime_dir: Path, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    meta_file = metadata.get("model_metadata")
    if not isinstance(meta_file, str):
        raise ValueError("Runtime metadata missing model_metadata")
    transformer_meta_path = runtime_dir / meta_file
    if not transformer_meta_path.exists():
        raise FileNotFoundError(
            f"Missing transformer metadata: {transformer_meta_path}"
        )
    meta = json.loads(transformer_meta_path.read_text(encoding="utf-8"))

    required = {
        "model_version",
        "feature_version",
        "feature_names",
        "tensor_contract",
        "trained_up_to_issue",
        "baseline_metrics",
        "expected_input_schema",
    }
    missing = sorted(required - set(meta.keys()))
    if missing:
        raise ValueError(f"Drift mismatch: missing transformer metadata keys {missing}")
    return meta


def _rank_top20(scores: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    return sorted(scores, key=lambda x: (-x["score"], x["number"]))[:20]


def _combo_metrics(
    combo: Tuple[Dict[str, float], ...], ranks: Dict[int, int]
) -> Tuple[int, int, int, int]:
    numbers = [int(item["number"]) for item in combo]
    tail_unique = len({n % 10 for n in numbers})
    cross_zone = (
        1 if any(n <= 40 for n in numbers) and any(n >= 41 for n in numbers) else 0
    )
    sorted_nums = sorted(numbers)
    adjacency = sum(1 for a, b in zip(sorted_nums, sorted_nums[1:]) if b - a == 1)
    rank_sum = sum(ranks[n] for n in numbers)
    return (tail_unique, cross_zone, -adjacency, -rank_sum)


def _select_top3(
    top20: Sequence[Dict[str, float]],
) -> Tuple[List[Dict[str, float]], bool]:
    ranks = {int(item["number"]): i for i, item in enumerate(top20)}
    combos = list(itertools.combinations(top20, 3))
    strict = [
        combo
        for combo in combos
        if _combo_metrics(combo, ranks)[0] == 3
        and _combo_metrics(combo, ranks)[1] == 1
        and _combo_metrics(combo, ranks)[2] == 0
    ]
    if strict:
        best = max(strict, key=lambda c: _combo_metrics(c, ranks))
        return sorted(best, key=lambda item: ranks[int(item["number"])]), False
    relaxed = max(combos, key=lambda c: _combo_metrics(c, ranks))
    return sorted(relaxed, key=lambda item: ranks[int(item["number"])]), True


def _to_issue(value: Any) -> int:
    return int(str(value))


def predict(runtime_dir: Path | None = None) -> Dict[str, object]:
    cfg = _load_predict_config()
    runtime_cfg = cfg.get("runtime")
    fetch_cfg = cfg.get("fetch")
    model_cfg = cfg.get("model")
    sources = cfg.get("auto_fetch_sources")

    if (
        not isinstance(runtime_cfg, dict)
        or not isinstance(fetch_cfg, dict)
        or not isinstance(model_cfg, dict)
    ):
        raise ValueError("Predict config missing runtime/fetch/model sections")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Predict config missing auto_fetch_sources")

    resolved_runtime = runtime_dir or Path(str(runtime_cfg.get("runtime_dir", "")))
    local_history_path = Path(str(runtime_cfg.get("local_history_path", "")))
    if not str(local_history_path):
        raise ValueError("runtime.local_history_path is required")

    seed = int(model_cfg.get("seed", 42))
    _set_deterministic(seed)

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

    fetched_latest_issue = _to_issue(latest_df.iloc[-1]["issue"])
    local_latest_issue = _to_issue(local_df.iloc[-1]["issue"])
    if fetched_latest_issue < local_latest_issue:
        raise ValueError(
            "Time-sync mismatch: fetched "
            f"{fetched_latest_issue} < local {local_latest_issue}"
        )

    merged = merge_history(local_df, latest_df)

    runtime_metadata = _load_runtime_metadata(resolved_runtime)
    transformer_meta = _load_transformer_metadata(resolved_runtime, runtime_metadata)

    if runtime_metadata.get("model_version") != model_cfg.get("model_version"):
        raise ValueError("Version mismatch: model_version")
    if runtime_metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature contract mismatch: runtime feature_version")
    if transformer_meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Drift mismatch: transformer feature_version")
    if transformer_meta.get("feature_names") != runtime_metadata.get("feature_names"):
        raise ValueError("Feature contract mismatch: feature_names")
    if transformer_meta.get("tensor_contract") != TENSOR_CONTRACT:
        raise ValueError("Tensor contract mismatch")

    window_size = int(model_cfg.get("window_size", 100))
    window = build_inference_window(merged, window_size)
    if window.feature_names != transformer_meta.get("feature_names"):
        raise ValueError("Feature contract mismatch: inference window feature_names")

    model_artifact = runtime_metadata.get("model_artifact")
    if not isinstance(model_artifact, str):
        raise ValueError("Runtime metadata missing model_artifact")
    ckpt_path = resolved_runtime / model_artifact
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {ckpt_path}")

    model = SmallTransformerRanker.load(
        ckpt_path,
        TransformerConfig(feature_dim=len(window.feature_names)),
    )
    model.eval()

    x_tensor = torch.from_numpy(window.features).unsqueeze(0)
    with torch.no_grad():
        score_tensor = model.predict_scores(x_tensor).squeeze(0)

    scores = [
        {"number": float(int(n)), "score": float(s)}
        for n, s in zip(window.number_ids.tolist(), score_tensor.tolist())
    ]
    top20 = _rank_top20(scores)
    top3, diversity_relaxed = _select_top3(top20)

    trained_issue = _to_issue(transformer_meta["trained_up_to_issue"])
    stale_issues = max(0, _to_issue(window.issue) - trained_issue)
    stale_threshold = int(runtime_metadata.get("stale_threshold", 20))

    return {
        "latest_known_issue": window.issue,
        "target_issue": window.target_issue,
        "model_version": runtime_metadata["model_version"],
        "feature_version": runtime_metadata["feature_version"],
        "data_source": data_source,
        "fetch_attempts": fetch_attempts,
        "score_type": "ranking_score",
        "scores": scores,
        "top20": top20,
        "top3": top3,
        "diversity_relaxed": diversity_relaxed,
        "drift_metadata": {
            "trained_up_to_issue": transformer_meta["trained_up_to_issue"],
            "baseline_metrics": transformer_meta["baseline_metrics"],
            "feature_version": transformer_meta["feature_version"],
            "expected_input_schema": transformer_meta["expected_input_schema"],
        },
        "stale_issues": stale_issues,
        "is_stale": stale_issues > stale_threshold,
    }
