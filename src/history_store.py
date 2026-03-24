"""Local history loading and merge utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

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
        normalized = df.rename(columns=RAW_COLUMN_MAP)
        return normalized[CANONICAL_COLUMNS].copy()

    raise ValueError("Local history schema mismatch")


def _resolve_history_path(local_history_path: Path) -> Path:
    if local_history_path.suffix.lower() == ".csv":
        parquet_path = local_history_path.with_suffix(".parquet")
        if parquet_path.exists():
            return parquet_path
    return local_history_path


def load_local_history(local_history_path: Path) -> pd.DataFrame:
    resolved_path = _resolve_history_path(local_history_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing local history: {resolved_path}")

    if resolved_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(resolved_path)
    else:
        df = pd.read_csv(resolved_path)

    normalized = _normalize_local_schema(df)
    if normalized.empty:
        raise ValueError("Local history is empty")

    normalized["issue"] = normalized["issue"].astype(str)
    for idx in range(1, 21):
        col = f"n{idx}"
        normalized[col] = pd.to_numeric(normalized[col], errors="raise").astype(int)
    normalized = normalized.sort_values(["issue"], kind="mergesort").reset_index(
        drop=True
    )
    return normalized


def merge_history(local_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
    local_issues = set(local_df["issue"].astype(str).tolist())

    for row in latest_df.itertuples(index=False):
        issue = str(row.issue)
        if issue not in local_issues:
            continue

        conflict_cols: List[str] = ["draw_time", *[f"n{i}" for i in range(1, 21)]]
        local_row = local_df[local_df["issue"].astype(str) == issue].iloc[-1]
        for col in conflict_cols:
            if str(local_row[col]) != str(getattr(row, col)):
                raise ValueError(
                    f"Issue conflict detected for issue={issue}, column={col}"
                )

    merged = pd.concat([local_df, latest_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["issue"], keep="first")
    merged = merged.sort_values(["issue"], kind="mergesort").reset_index(drop=True)
    return merged
