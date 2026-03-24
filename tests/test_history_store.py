from pathlib import Path

import pandas as pd
import pytest

from src.history_store import load_local_history, merge_history


def test_merge_history_conflict_fail(tmp_path: Path) -> None:
    local_path = tmp_path / "history.csv"
    local = pd.DataFrame(
        [
            {
                "issue": "1001",
                "draw_time": "2026-01-01",
                **{f"n{i}": i for i in range(1, 21)},
            }
        ]
    )
    local.to_csv(local_path, index=False)
    local_df = load_local_history(local_path)

    latest_df = pd.DataFrame(
        [
            {
                "issue": "1001",
                "draw_time": "2026-01-02",
                **{f"n{i}": i for i in range(1, 21)},
            }
        ]
    )

    with pytest.raises(ValueError, match="Issue conflict"):
        merge_history(local_df, latest_df)


def test_merge_history_dedupe_success(tmp_path: Path) -> None:
    local_path = tmp_path / "history.csv"
    local = pd.DataFrame(
        [
            {
                "issue": "1001",
                "draw_time": "2026-01-01",
                **{f"n{i}": i for i in range(1, 21)},
            }
        ]
    )
    local.to_csv(local_path, index=False)
    local_df = load_local_history(local_path)

    latest_df = pd.DataFrame(
        [
            {
                "issue": "1002",
                "draw_time": "2026-01-02",
                **{f"n{i}": i + 1 for i in range(1, 21)},
            }
        ]
    )

    merged = merge_history(local_df, latest_df)
    if merged["issue"].tolist() != ["1001", "1002"]:
        pytest.fail("merged issues mismatch")
