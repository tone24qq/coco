from pathlib import Path

import pandas as pd
import pytest

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


def test_runtime_history_build_with_model_source(tmp_path: Path) -> None:
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
        stale_threshold=20,
    )

    runtime_dir = tmp_path / "runtime"
    build_runtime_history(input_path, runtime_dir, model_source)

    required = [
        "history_runtime.parquet",
        "history_runtime.csv",
        "scores.parquet",
        "scores.csv",
        "model.ckpt",
        "transformer_metadata.json",
        "metadata.json",
    ]
    for name in required:
        if not (runtime_dir / name).exists():
            pytest.fail(f"missing runtime artifact: {name}")


def test_runtime_history_missing_model_fail(tmp_path: Path) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path)
    with pytest.raises(FileNotFoundError, match="Missing model artifacts"):
        build_runtime_history(
            input_path, tmp_path / "runtime", tmp_path / "empty_models"
        )


def test_runtime_history_model_size_limit_fail(tmp_path: Path) -> None:
    input_path = tmp_path / "history.csv"
    _make_history(input_path)
    model_source = tmp_path / "models"
    model_source.mkdir(parents=True, exist_ok=True)
    oversized = model_source / "model.ckpt"
    oversized.write_bytes(b"0")
    oversized.touch()
    with oversized.open("r+b") as fh:
        fh.truncate((100 * 1024 * 1024) + 1)
    (model_source / "transformer_metadata.json").write_text(
        '{"model_version":"small_transformer_v2","feature_version":"rank_window_v2",'
        '"feature_names":[],"tensor_contract":{},'
        '"trained_up_to_issue":"1000","baseline_metrics":{},'
        '"expected_input_schema":[],"expected_output_schema":[],"stale_threshold":20}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Model artifact too large"):
        build_runtime_history(input_path, tmp_path / "runtime", model_source)
