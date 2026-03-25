import json
import time
from pathlib import Path

import pandas as pd
import pytest

from src.inference import predict
from src.runtime_history import build_runtime_history
from src.train_transformer import train_model


def _make_history(path: Path) -> None:
    rows = []
    for issue in range(1000, 1065):
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
        window_size=20,
        seed=42,
        epochs=1,
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
            "window_size": 20,
            "seed": 42,
            "stale_threshold": 3,
        },
        "tensor_contract": {
            "raw_tensor": "[batch, 80, feature_dim]",
            "model_input_tensor": "[batch, 80, d_model]",
            "attention_axis": "candidate-to-candidate",
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
            [{"issue": "1065", "draw_time": "x", "numbers": list(range(1, 21))}],
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
            [{"issue": "1065", "draw_time": "x", "numbers": list(range(1, 21))}],
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
            [{"issue": "1060", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )
    with pytest.raises(ValueError, match="Time-sync mismatch"):
        predict(runtime_dir)


def test_tensor_contract_drift_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)

    meta_path = runtime_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tensor_contract"] = {"raw_tensor": "bad"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1065", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )
    with pytest.raises(ValueError, match="Tensor contract mismatch"):
        predict(runtime_dir)


def test_missing_model_artifact_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)
    (runtime_dir / "model.ckpt").unlink()

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1065", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [],
        ),
    )
    with pytest.raises(FileNotFoundError, match="Missing model artifact"):
        predict(runtime_dir)


def test_predict_outputs_chinese_timing_progress_and_sla(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1065", "draw_time": "x", "numbers": list(range(1, 21))}],
            "mock",
            [{"source": "mock", "attempt": 1, "status": "ok"}],
        ),
    )

    started = time.perf_counter()
    first = predict(runtime_dir)
    second = predict(runtime_dir)
    elapsed = time.perf_counter() - started
    out = capsys.readouterr().out

    if "[預測進度]" not in out or "總耗時" not in out:
        pytest.fail("predict should output chinese timing progress")
    if elapsed > 10.0:
        pytest.fail("predict total latency should be <= 10 seconds in this test")
    if first["top20"] != second["top20"] or first["top3"] != second["top3"]:
        pytest.fail("predict outputs changed after optimization path")


def test_predict_includes_source_diagnostics_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            [{"issue": "1065", "draw_time": "x", "numbers": list(range(1, 21))}],
            "auzo",
            [{"source": "winwin", "status": "ok"}, {"source": "auzo", "status": "ok"}],
            {
                "source_latest_issues": {"winwin": "1064", "auzo": "1065"},
                "selected_source_reason": "selected highest latest_issue",
                "source_records_count": {"winwin": 5, "auzo": 8},
                "consensus_status": "partial",
                "max_observed_issue": "1065",
                "source_consensus": {"latest_issue_gap": 1, "conflicts": []},
            },
        ),
    )

    result = predict(runtime_dir)
    if result["latest_known_issue"] != "1065":
        pytest.fail("latest_known_issue should match selected latest source issue")
    if result["source_latest_issues"] != {"winwin": "1064", "auzo": "1065"}:
        pytest.fail("source_latest_issues should be present in response")
    if result["selected_source_reason"] != "selected highest latest_issue":
        pytest.fail("selected_source_reason should be present in response")
    if result["source_records_count"] != {"winwin": 5, "auzo": 8}:
        pytest.fail("source_records_count should be present in response")
    if (
        result["consensus_status"] != "partial"
        or result["max_observed_issue"] != "1065"
    ):
        pytest.fail("consensus diagnostics should be present in response")


def test_predict_uses_full_day_records_latest_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, runtime_dir = _prepare_runtime(tmp_path)
    config_path = _write_config(tmp_path, input_path, runtime_dir)

    full_records = [
        {"issue": str(issue), "draw_time": "x", "numbers": list(range(1, 21))}
        for issue in range(1065, 1075)
    ]

    monkeypatch.setattr("src.inference.CONFIG_PATH", config_path)
    monkeypatch.setattr(
        "src.inference.fetch_latest",
        lambda sources, config: (
            full_records,
            "auzo",
            [{"source": "auzo", "status": "ok"}],
            {
                "source_latest_issues": {"auzo": "1074"},
                "selected_source_reason": "selected highest latest_issue",
                "source_records_count": {"auzo": 10},
                "source_tail_count": {"auzo": 5},
                "consensus_status": "unanimous",
                "max_observed_issue": "1074",
                "selected_source_full_records_count": 10,
                "selected_source_tail_count": 5,
                "source_consensus": {"latest_issue_gap": 0, "conflicts": []},
            },
        ),
    )

    result = predict(runtime_dir)
    if result["latest_known_issue"] != "1074":
        pytest.fail("latest_known_issue should be selected full record latest issue")
    if result["target_issue"] != "1075":
        pytest.fail("target_issue should be latest_known_issue + 1")
    if result["selected_source_full_records_count"] != 10:
        pytest.fail("response should expose full records count")
    if result["selected_source_tail_count"] != 5:
        pytest.fail("response should expose selected source tail count")
