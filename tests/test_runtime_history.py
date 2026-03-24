from pathlib import Path

import pandas as pd
import pytest

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


def test_runtime_history_build_with_model_source(tmp_path: Path) -> None:
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
