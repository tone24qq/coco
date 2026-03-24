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


def _validate_issue_sequence(df: pd.DataFrame, label: str) -> None:
    if df["issue"].duplicated().any():
        raise ValueError(f"{label}: duplicated issue values")

    issues = [int(x) for x in df["issue"].tolist()]
    if issues != sorted(issues):
        raise ValueError(f"{label}: issue not monotonic increasing")


def load_local_history(local_history_path: Path) -> pd.DataFrame:
    resolved = _resolve_history_path(local_history_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing local history: {resolved}")

    if resolved.suffix.lower() == ".parquet":
        raw = pd.read_parquet(resolved)
    else:
        raw = pd.read_csv(resolved)

    normalized = _normalize_local_schema(raw)
    if normalized.empty:
        raise ValueError("Local history is empty")

    normalized["issue"] = normalized["issue"].astype(str)
    for idx in range(1, 21):
        col = f"n{idx}"
        normalized[col] = pd.to_numeric(normalized[col], errors="raise").astype(int)

    normalized = normalized.sort_values(["issue"], kind="mergesort").reset_index(
        drop=True
    )
    _validate_issue_sequence(normalized, "local_history")
    return normalized


def merge_history(local_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
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
    merged = merged.sort_values(["issue"], kind="mergesort").reset_index(drop=True)
    _validate_issue_sequence(merged, "merged_history")

    merged_issues = [int(x) for x in merged["issue"].tolist()]
    if len(merged_issues) >= 2:
        for left, right in zip(merged_issues, merged_issues[1:]):
            if right - left <= 0:
                raise ValueError("Merged history issue continuity check failed")

    return merged
