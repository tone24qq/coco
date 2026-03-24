"""Local history loading and merge utilities with strict validators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.normalize_latest import CANONICAL_COLUMNS

RAW_COLUMN_MAP = {
    "期別": "issue",
    "開獎日期": "draw_time",
    **{f"獎號{i}": f"n{i}" for i in range(1, 21)},
}


def _normalize_local_schema(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in CANONICAL_COLUMNS):
        return df[CANONICAL_COLUMNS].copy()
    if all(col in df.columns for col in RAW_COLUMN_MAP):
        return df.rename(columns=RAW_COLUMN_MAP)[CANONICAL_COLUMNS].copy()
    raise ValueError("Local history schema mismatch")


def _resolve_history_path(local_history_path: Path) -> Path:
    if local_history_path.suffix.lower() == ".csv":
        parquet_path = local_history_path.with_suffix(".parquet")
        if parquet_path.exists():
            return parquet_path
    return local_history_path


def _coerce_and_validate_numbers(df: pd.DataFrame, label: str) -> pd.DataFrame:
    normalized = df.copy()
    normalized["issue"] = normalized["issue"].astype(str)
    for idx in range(1, 21):
        col = f"n{idx}"
        normalized[col] = pd.to_numeric(normalized[col], errors="raise").astype(int)
        if ((normalized[col] < 1) | (normalized[col] > 80)).any():
            raise ValueError(f"{label}: {col} contains values outside 1..80")
    return normalized


def _validate_issue_sequence(df: pd.DataFrame, label: str) -> None:
    if df["issue"].duplicated().any():
        raise ValueError(f"{label}: duplicated issue values")

    issues = [int(x) for x in df["issue"].tolist()]
    if issues != sorted(issues):
        raise ValueError(f"{label}: issue not monotonic increasing")


def _validate_consecutive_issues(df: pd.DataFrame, label: str) -> None:
    issues = [int(x) for x in df["issue"].tolist()]
    if len(issues) < 2:
        return
    for left, right in zip(issues, issues[1:]):
        if right - left != 1:
            raise ValueError(f"{label}: issues are not consecutive")


def load_local_history(local_history_path: Path) -> pd.DataFrame:
    resolved = _resolve_history_path(local_history_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing local history: {resolved}")

    if resolved.suffix.lower() == ".parquet":
        raw = pd.read_parquet(resolved)
    else:
        raw = pd.read_csv(resolved)

    normalized = _normalize_local_schema(raw)
    normalized = _coerce_and_validate_numbers(normalized, "local_history")
    if normalized.empty:
        raise ValueError("Local history is empty")

    if (
        resolved.suffix.lower() == ".parquet"
        and local_history_path.suffix.lower() == ".csv"
    ):
        if local_history_path.exists():
            csv_raw = pd.read_csv(local_history_path)
            csv_normalized = _coerce_and_validate_numbers(
                _normalize_local_schema(csv_raw), "local_history_csv"
            )
            if sorted(csv_normalized.columns.tolist()) != sorted(
                normalized.columns.tolist()
            ):
                raise ValueError(
                    "Local history schema mismatch between parquet and csv"
                )

    normalized["issue_int"] = normalized["issue"].astype(int)
    normalized = normalized.sort_values(["issue_int"], kind="mergesort").drop(
        columns=["issue_int"]
    )
    normalized = normalized.reset_index(drop=True)
    _validate_issue_sequence(normalized, "local_history")
    return normalized


def merge_history(local_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
    if sorted(local_df.columns.tolist()) != sorted(CANONICAL_COLUMNS):
        raise ValueError("local_history: schema mismatch")
    if sorted(latest_df.columns.tolist()) != sorted(CANONICAL_COLUMNS):
        raise ValueError("latest_history: schema mismatch")

    _validate_issue_sequence(local_df, "local_history")
    _validate_issue_sequence(latest_df, "latest_history")

    local_index = {str(row.issue): row for row in local_df.itertuples(index=False)}
    for latest in latest_df.itertuples(index=False):
        issue = str(latest.issue)
        if issue not in local_index:
            continue
        local_row = local_index[issue]
        for col in ["draw_time", *[f"n{i}" for i in range(1, 21)]]:
            if str(getattr(local_row, col)) != str(getattr(latest, col)):
                raise ValueError(
                    f"Issue conflict detected for issue={issue}, column={col}"
                )

    merged = pd.concat([local_df, latest_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["issue"], keep="first")
    merged["issue_int"] = merged["issue"].astype(int)
    merged = merged.sort_values(["issue_int"], kind="mergesort").drop(
        columns=["issue_int"]
    )
    merged = merged.reset_index(drop=True)
    _validate_issue_sequence(merged, "merged_history")
    _validate_consecutive_issues(merged, "merged_history")

    return merged
