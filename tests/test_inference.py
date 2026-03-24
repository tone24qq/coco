import json
from pathlib import Path

import pandas as pd
import pytest

from src.inference import predict
from src.runtime_history import build_runtime_history


def _prepare_runtime(tmp_path: Path) -> Path:
    history_path = tmp_path / "history.csv"
    rows = []
    for issue in range(1000, 1120):
        rows.append(
            {
                "issue": issue,
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    pd.DataFrame(rows).to_csv(history_path, index=False)
    runtime_dir = tmp_path / "runtime"
    build_runtime_history(history_path, runtime_dir)
    return runtime_dir


def _write_config(tmp_path: Path, runtime_dir: Path, local_path: Path) -> Path:
    cfg = {
        "auto_fetch_sources": [{"name": "mock", "url": "https://mock"}],
        "fetch": {"timeout_seconds": 3.0, "retries": 0, "backoff_seconds": 0.0},
        "runtime": {
            "local_history_path": str(local_path),
            "runtime_dir": str(runtime_dir),
        },
        "model": {
            "artifact_file": "transformer_model.npz",
            "model_version": "small_transformer_v1",
            "feature_version": "rank_window_v1",
            "window_size": 100,
            "seed": 42,
        },
    }
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    return config_path


def test_predict_success_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = _prepare_runtime(tmp_path)
    local_path = tmp_path / "local.csv"
    pd.read_csv(runtime_dir / "history_runtime.csv").to_csv(local_path, index=False)

    config_path = _write_config(tmp_path, runtime_dir, local_path)
    latest = [
        {
            "issue": "1120",
            "draw_time": "2026-01-02",
            "numbers": list(range(1, 21)),
        }
    ]
    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (latest, "mock_source", [{"status": "ok"}]),
    )

    first = predict()
    second = predict()
    if first["top20"] != second["top20"] or first["top3"] != second["top3"]:
        pytest.fail("predict output must be deterministic")
    if "diversity_relaxed" not in first:
        pytest.fail("predict must include diversity_relaxed metadata")


def test_predict_time_sync_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = _prepare_runtime(tmp_path)
    local_path = tmp_path / "local.csv"
    pd.read_csv(runtime_dir / "history_runtime.csv").to_csv(local_path, index=False)

    config_path = _write_config(tmp_path, runtime_dir, local_path)
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
        predict()


def test_predict_version_mismatch_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = _prepare_runtime(tmp_path)
    metadata_path = runtime_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_version"] = "mismatch"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    local_path = tmp_path / "local.csv"
    pd.read_csv(runtime_dir / "history_runtime.csv").to_csv(local_path, index=False)

    config_path = _write_config(tmp_path, runtime_dir, local_path)
    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1120", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )

    with pytest.raises(ValueError, match="Model version mismatch"):
        predict()


def test_predict_missing_artifact_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = _prepare_runtime(tmp_path)
    (runtime_dir / "transformer_model.npz").unlink()

    local_path = tmp_path / "local.csv"
    pd.read_csv(runtime_dir / "history_runtime.csv").to_csv(local_path, index=False)

    config_path = _write_config(tmp_path, runtime_dir, local_path)
    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1120", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )

    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        predict()
