from pathlib import Path

import pandas as pd
import pytest

from src.history_store import load_local_history, merge_history


def _row(issue: str, offset: int) -> dict:
    return {
        "issue": issue,
        "draw_time": "2026-01-01",
        **{f"n{i}": ((offset + i) % 80) + 1 for i in range(1, 21)},
    }


def test_duplicate_issue_fail(tmp_path: Path) -> None:
    p = tmp_path / "history.csv"
    pd.DataFrame([_row("1001", 1), _row("1001", 2)]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicated issue"):
        load_local_history(p)


def test_conflict_issue_fail(tmp_path: Path) -> None:
    p = tmp_path / "history.csv"
    pd.DataFrame([_row("1001", 1)]).to_csv(p, index=False)
    local = load_local_history(p)
    latest = pd.DataFrame([_row("1001", 9)])
    with pytest.raises(ValueError, match="Issue conflict"):
        merge_history(local, latest)


def test_parquet_preferred(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    parquet_path = tmp_path / "history.parquet"
    pd.DataFrame([_row("1001", 1)]).to_csv(csv_path, index=False)
    pd.DataFrame([_row("1002", 2)]).to_parquet(parquet_path, index=False)
    loaded = load_local_history(csv_path)
    if loaded.iloc[-1]["issue"] != "1002":
        pytest.fail("parquet preference failed")


def test_merge_allows_missing_issues() -> None:
    local = pd.DataFrame([_row("1001", 1), _row("1003", 3)])
    latest = pd.DataFrame([_row("1004", 4)])
    merged = merge_history(local, latest)
    assert merged["issue"].tolist() == ["1001", "1003", "1004"]
