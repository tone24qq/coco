from pathlib import Path

import pandas as pd
import pytest

from src.runtime_history import build_runtime_history


@pytest.fixture
def sample_input(tmp_path: Path) -> Path:
    rows = []
    for issue in range(1000, 1120):
        rows.append(
            {
                "issue": issue,
                "draw_time": "2026-01-01T00:00:00",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    input_path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    return input_path


def test_build_runtime_history_success(sample_input: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "runtime"
    build_runtime_history(sample_input, output_dir)

    required = [
        "metadata.json",
        "transformer_metadata.json",
        "transformer_model.npz",
        "scores.parquet",
        "scores.csv",
        "history_runtime.parquet",
        "history_runtime.csv",
    ]
    for file_name in required:
        if not (output_dir / file_name).exists():
            pytest.fail(f"missing runtime artifact: {file_name}")


def test_build_runtime_history_schema_mismatch(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame([{"issue": 1, "x": 2}]).to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="Input schema mismatch"):
        build_runtime_history(bad_path, tmp_path / "runtime")


def test_build_runtime_history_missing_input_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        build_runtime_history(tmp_path / "missing.csv", tmp_path / "runtime")
