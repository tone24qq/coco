from pathlib import Path

import pandas as pd
import pytest

from src.runtime_history import build_runtime_history


@pytest.fixture
def sample_input(tmp_path: Path) -> Path:
    rows = [
        {
            "issue": 1001,
            "draw_time": "2026-01-01T00:00:00",
            **{f"n{i}": i for i in range(1, 21)},
        },
        {
            "issue": 1002,
            "draw_time": "2026-01-01T00:05:00",
            **{f"n{i}": i + 1 for i in range(1, 21)},
        },
    ]
    input_path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    return input_path


def test_build_runtime_history_success(sample_input: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "runtime"
    build_runtime_history(sample_input, output_dir)

    metadata_path = output_dir / "metadata.json"
    scores_path = output_dir / "scores.csv"

    if not metadata_path.exists():
        pytest.fail("metadata.json was not generated")
    if not scores_path.exists():
        pytest.fail("scores.csv was not generated")


def test_build_runtime_history_schema_mismatch(tmp_path: Path) -> None:
    input_path = tmp_path / "bad_history.csv"
    bad_row = {"issue": 1, **{f"x{i}": i for i in range(1, 4)}}
    pd.DataFrame([bad_row]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Input schema mismatch"):
        build_runtime_history(input_path, tmp_path / "runtime")


def test_build_runtime_history_missing_input_fail(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        build_runtime_history(missing_path, tmp_path / "runtime")
