"""Runtime inference pipeline using trained PyTorch transformer checkpoint."""

from __future__ import annotations

import itertools
import json
import random
import time
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
_CONFIG_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RUNTIME_METADATA_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_TRANSFORMER_METADATA_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_LOCAL_HISTORY_CACHE: Dict[str, Tuple[float, Any]] = {}
_MODEL_CACHE: Dict[str, Tuple[float, SmallTransformerRanker]] = {}


def _progress(percent: int, message: str, elapsed_seconds: float) -> None:
    print(
        f"[預測進度] {percent}% {message}（{elapsed_seconds:.2f} 秒）",
        flush=True,
    )


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


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
    key = str(config_path.resolve())
    mtime = _mtime(config_path)
    cached = _CONFIG_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return dict(cached[1])
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Predict config schema mismatch")
    _CONFIG_CACHE[key] = (mtime, dict(cfg))
    return cfg


def _load_runtime_metadata(runtime_dir: Path) -> Dict[str, Any]:
    meta_path = runtime_dir / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata artifact: {meta_path}")
    key = str(meta_path.resolve())
    mtime = _mtime(meta_path)
    cached = _RUNTIME_METADATA_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        metadata = dict(cached[1])
    else:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _RUNTIME_METADATA_CACHE[key] = (mtime, dict(metadata))
    required = {
        "model_version",
        "feature_version",
        "feature_names",
        "tensor_contract",
        "model_artifact",
        "model_metadata",
        "expected_input_schema",
        "expected_output_schema",
        "stale_threshold",
    }
    missing = sorted(required - set(metadata.keys()))
    if missing:
        raise ValueError(f"Runtime metadata mismatch: missing keys {missing}")
    return metadata


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
    key = str(transformer_meta_path.resolve())
    mtime = _mtime(transformer_meta_path)
    cached = _TRANSFORMER_METADATA_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        meta = dict(cached[1])
    else:
        meta = json.loads(transformer_meta_path.read_text(encoding="utf-8"))
        _TRANSFORMER_METADATA_CACHE[key] = (mtime, dict(meta))

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


def _validate_expected_output_schema(runtime_metadata: Dict[str, Any]) -> None:
    expected = runtime_metadata.get("expected_output_schema")
    if not isinstance(expected, list):
        raise ValueError("Runtime metadata mismatch: expected_output_schema")
    required = {
        "latest_known_issue",
        "target_issue",
        "model_version",
        "feature_version",
        "data_source",
        "scores",
        "top20",
        "top3",
    }
    if not required.issubset(set(str(item) for item in expected)):
        raise ValueError("Runtime metadata mismatch: expected_output_schema")


def predict(runtime_dir: Path | None = None) -> Dict[str, object]:
    total_start = time.perf_counter()
    stage_start = total_start
    cfg = _load_predict_config()
    _progress(12, "載入 config 完成", time.perf_counter() - stage_start)
    runtime_cfg = cfg.get("runtime")
    fetch_cfg = cfg.get("fetch")
    model_cfg = cfg.get("model")
    tensor_cfg = cfg.get("tensor_contract")
    sources = cfg.get("auto_fetch_sources")

    if (
        not isinstance(runtime_cfg, dict)
        or not isinstance(fetch_cfg, dict)
        or not isinstance(model_cfg, dict)
        or not isinstance(tensor_cfg, dict)
    ):
        raise ValueError("Predict config missing runtime/fetch/model/tensor sections")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Predict config missing auto_fetch_sources")

    resolved_runtime = runtime_dir or Path(str(runtime_cfg.get("runtime_dir", "")))
    local_history_path = Path(str(runtime_cfg.get("local_history_path", "")))
    if not str(local_history_path):
        raise ValueError("runtime.local_history_path is required")

    seed = int(model_cfg.get("seed", 42))
    _set_deterministic(seed)

    stage_start = time.perf_counter()
    fetch_result = fetch_latest(
        sources=sources,
        config=FetchConfig(
            timeout_seconds=float(fetch_cfg.get("timeout_seconds", 8.0)),
            retries=int(fetch_cfg.get("retries", 2)),
            backoff_seconds=float(fetch_cfg.get("backoff_seconds", 0.5)),
        ),
    )
    if len(fetch_result) == 3:
        latest_records, data_source, fetch_attempts = fetch_result
        fetch_diagnostics: Dict[str, object] = {}
    else:
        latest_records, data_source, fetch_attempts, fetch_diagnostics = fetch_result
    _progress(30, "抓取最新資料完成", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    latest_df = normalize_latest_records(latest_records)
    _progress(38, "最新資料標準化完成", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    local_key = str(local_history_path.resolve())
    local_mtime = _mtime(local_history_path)
    local_cached = _LOCAL_HISTORY_CACHE.get(local_key)
    if local_cached is not None and local_cached[0] == local_mtime:
        local_df = local_cached[1].copy()
    else:
        local_df = load_local_history(local_history_path)
        _LOCAL_HISTORY_CACHE[local_key] = (local_mtime, local_df.copy())
    _progress(46, "載入本機歷史完成", time.perf_counter() - stage_start)

    fetched_latest_issue = _to_issue(latest_df.iloc[-1]["issue"])
    local_latest_issue = _to_issue(local_df.iloc[-1]["issue"])
    if fetched_latest_issue < local_latest_issue:
        raise ValueError(
            "Time-sync mismatch: fetched "
            f"{fetched_latest_issue} < local {local_latest_issue}"
        )

    stage_start = time.perf_counter()
    merged = merge_history(local_df, latest_df)
    _progress(56, "合併歷史與時序檢查完成", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    runtime_metadata = _load_runtime_metadata(resolved_runtime)
    _validate_expected_output_schema(runtime_metadata)
    transformer_meta = _load_transformer_metadata(resolved_runtime, runtime_metadata)
    _progress(64, "載入 runtime artifacts 完成", time.perf_counter() - stage_start)

    if runtime_metadata.get("model_version") != model_cfg.get("model_version"):
        raise ValueError("Version mismatch: model_version")
    if runtime_metadata.get("feature_version") != model_cfg.get("feature_version"):
        raise ValueError("Version mismatch: feature_version")
    if runtime_metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Feature contract mismatch: runtime feature_version")
    if transformer_meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Drift mismatch: transformer feature_version")
    if runtime_metadata.get("tensor_contract") != tensor_cfg:
        raise ValueError("Tensor contract mismatch: config vs runtime metadata")
    if transformer_meta.get("feature_names") != runtime_metadata.get("feature_names"):
        raise ValueError("Feature contract mismatch: feature_names")
    if transformer_meta.get("tensor_contract") != TENSOR_CONTRACT:
        raise ValueError("Tensor contract mismatch")

    window_size = int(model_cfg.get("window_size", 100))
    stage_start = time.perf_counter()
    window = build_inference_window(merged, window_size)
    _progress(74, "建立 inference window 完成", time.perf_counter() - stage_start)
    if window.feature_names != transformer_meta.get("feature_names"):
        raise ValueError("Feature contract mismatch: inference window feature_names")

    model_artifact = runtime_metadata.get("model_artifact")
    if not isinstance(model_artifact, str):
        raise ValueError("Runtime metadata missing model_artifact")
    ckpt_path = resolved_runtime / model_artifact
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {ckpt_path}")

    stage_start = time.perf_counter()
    model_key = str(ckpt_path.resolve())
    model_mtime = _mtime(ckpt_path)
    cached_model = _MODEL_CACHE.get(model_key)
    if cached_model is not None and cached_model[0] == model_mtime:
        model = cached_model[1]
    else:
        model = SmallTransformerRanker.load(
            ckpt_path,
            TransformerConfig(feature_dim=len(window.feature_names)),
        )
        _MODEL_CACHE[model_key] = (model_mtime, model)
    model.eval()
    _progress(82, "載入模型完成", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    x_tensor = torch.from_numpy(window.features).unsqueeze(0)
    with torch.no_grad():
        score_tensor = model.predict_scores(x_tensor).squeeze(0)
    _progress(88, "模型推論完成", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    scores = [
        {"number": int(n), "score": float(s)}
        for n, s in zip(window.number_ids.tolist(), score_tensor.tolist())
    ]
    top20 = _rank_top20(scores)
    top3, diversity_relaxed = _select_top3(top20)
    _progress(94, "Top20 / Top3 重排完成", time.perf_counter() - stage_start)

    trained_issue = _to_issue(transformer_meta["trained_up_to_issue"])
    stale_issues = max(0, _to_issue(window.issue) - trained_issue)
    stale_threshold = int(runtime_metadata.get("stale_threshold", 20))

    result = {
        "latest_known_issue": window.issue,
        "target_issue": window.target_issue,
        "model_version": runtime_metadata["model_version"],
        "feature_version": runtime_metadata["feature_version"],
        "data_source": data_source,
        "fetch_attempts": fetch_attempts,
        "source_latest_issues": fetch_diagnostics.get("source_latest_issues", {}),
        "selected_source_reason": fetch_diagnostics.get("selected_source_reason", ""),
        "source_records_count": fetch_diagnostics.get("source_records_count", {}),
        "source_tail_count": fetch_diagnostics.get("source_tail_count", {}),
        "consensus_status": fetch_diagnostics.get("consensus_status", "partial"),
        "max_observed_issue": fetch_diagnostics.get(
            "max_observed_issue", str(window.issue)
        ),
        "selected_source_full_records_count": fetch_diagnostics.get(
            "selected_source_full_records_count", len(latest_records)
        ),
        "selected_source_tail_count": fetch_diagnostics.get(
            "selected_source_tail_count", len(latest_records)
        ),
        "source_consensus": fetch_diagnostics.get("source_consensus", {}),
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
    total_elapsed = time.perf_counter() - total_start
    _progress(100, f"預測完成，總耗時 {total_elapsed:.2f}", total_elapsed)
    return result
