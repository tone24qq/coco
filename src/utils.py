from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


class DataContractError(ValueError):
    """Raised when data violates project schema."""


@dataclass(frozen=True)
class DrawRecord:
    issue: str
    draw_date: date
    numbers: tuple[int, ...]
    day_issue_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "draw_date": self.draw_date.isoformat(),
            "numbers": list(self.numbers),
            "day_issue_index": self.day_issue_index,
        }


def ensure_numbers(numbers: Iterable[int]) -> tuple[int, ...]:
    arr = tuple(sorted(int(n) for n in numbers))
    if len(arr) != 20:
        raise DataContractError(f"each draw must contain 20 numbers, got {len(arr)}")
    if len(set(arr)) != 20:
        raise DataContractError("draw numbers must be unique")
    if arr[0] < 1 or arr[-1] > 80:
        raise DataContractError("draw numbers must be in range 1..80")
    return arr


def parse_date(text: str) -> date:
    raw = text.strip()
    candidates = [raw, raw.replace("/", "-")]
    for c in candidates:
        try:
            return datetime.strptime(c, "%Y-%m-%d").date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(candidates[-1]).date()
    except ValueError as exc:
        raise DataContractError(f"unsupported date format: {text}") from exc


def read_processed(path: Path) -> list[DrawRecord]:
    rows: list[DrawRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"issue", "draw_date", "numbers", "day_issue_index"}
        if not required.issubset(reader.fieldnames or []):
            raise DataContractError(f"processed CSV missing columns: {required}")
        for row in reader:
            rows.append(
                DrawRecord(
                    issue=row["issue"],
                    draw_date=parse_date(row["draw_date"]),
                    numbers=ensure_numbers(json.loads(row["numbers"])),
                    day_issue_index=int(row["day_issue_index"]),
                )
            )
    return rows


def write_processed(path: Path, records: list[DrawRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["issue", "draw_date", "numbers", "day_issue_index"],
        )
        writer.writeheader()
        for record in records:
            item = record.to_dict()
            item["numbers"] = json.dumps(item["numbers"], ensure_ascii=False)
            writer.writerow(item)


def rolling_hit_count(history: list[DrawRecord], number: int, window: int) -> int:
    span = history[-window:] if window > 0 else history
    return sum(1 for row in span if number in row.numbers)
