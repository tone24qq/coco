"""Normalize and validate externally fetched latest draw records."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

CANONICAL_COLUMNS = ["issue", "draw_time", *[f"n{i}" for i in range(1, 21)]]


def _validate_row(record: Dict[str, object]) -> Dict[str, object]:
    if "issue" not in record or "draw_time" not in record or "numbers" not in record:
        raise ValueError(
            "Latest record schema mismatch: issue/draw_time/numbers required"
        )

    numbers = record["numbers"]
    if not isinstance(numbers, list) or len(numbers) != 20:
        raise ValueError("Each latest record must contain exactly 20 numbers")

    normalized_numbers: List[int] = []
    for number in numbers:
        try:
            n_value = int(number)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid number value in latest record: {number}"
            ) from exc
        if n_value < 1 or n_value > 80:
            raise ValueError(f"Latest number out of range 1..80: {n_value}")
        normalized_numbers.append(n_value)

    if len(set(normalized_numbers)) != 20:
        raise ValueError("Latest record contains duplicate numbers")

    normalized: Dict[str, object] = {
        "issue": str(record["issue"]),
        "draw_time": str(record["draw_time"]),
    }
    for idx, value in enumerate(normalized_numbers, start=1):
        normalized[f"n{idx}"] = value
    return normalized


def normalize_latest_records(records: List[Dict[str, object]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No latest records fetched")

    normalized_rows = [_validate_row(record) for record in records]
    df = pd.DataFrame(normalized_rows)
    if sorted(df.columns.tolist()) != sorted(CANONICAL_COLUMNS):
        raise ValueError("Latest normalized schema mismatch")

    if df["issue"].duplicated().any():
        raise ValueError("Latest records contain duplicate issue values")

    df = df[CANONICAL_COLUMNS].copy()
    df = df.sort_values(["issue"], kind="mergesort").reset_index(drop=True)
    return df
