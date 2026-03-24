import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference import predict
from src.runtime_history import build_runtime_history
from src.train_transformer import train_model


def _make_history(path: Path) -> None:
    rows = []
    for issue in range(1000, 1120):
        rows.append(
            {
                "issue": issue,
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _prepare_runtime(tmp_path: Path) -> tuple[Path, Path]:
    input_path = tmp_path / "history.csv"
    _make_history(input_path)
    model_source = tmp_path / "models"
    train_model(
        input_path=input_path,
        output_dir=model_source,
        model_file="model.ckpt",
        window_size=50,
        seed=42,
        epochs=2,
        batch_size=16,
        alpha=0.2,
        stale_threshold=3,
    )
    runtime_dir = tmp_path / "runtime"
    build_runtime_history(input_path, runtime_dir, model_source)
    return input_path, runtime_dir


def _write_config(tmp_path: Path, local_history: Path, runtime_dir: Path) -> Path:
    cfg = {
        "auto_fetch_sources": [{"name": "mock", "url": "https://mock"}],
        "fetch": {"timeout_seconds": 1.0, "retries": 0, "backoff_seconds": 0.0},
        "runtime": {
            "local_history_path": str(local_history),
            "runtime_dir": str(runtime_dir),
        },
        "model": {
            "artifact_file": "model.ckpt",
            "model_version": "small_transformer_v2",
            "feature_version": "rank_window_v2",
            "window_size": 50,
            "seed": 42,
            "stale_threshold": 3,
        },
    }
    p = tmp_path / "predict.yaml"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_inference_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1120", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [{"status": "ok"}],
        ),
    )

    one = predict(runtime_dir)
    two = predict(runtime_dir)
    if one["top20"] != two["top20"] or one["top3"] != two["top3"]:
        pytest.fail("inference must be deterministic")


def test_feature_names_drift_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)

    meta_path = runtime_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["feature_names"] = ["bad"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1120", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )

    with pytest.raises(ValueError, match="feature_names"):
        predict(runtime_dir)


def test_time_sync_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)
    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1110", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )
    with pytest.raises(ValueError, match="Time-sync mismatch"):
        predict(runtime_dir)
