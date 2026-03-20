from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class DataContractError(ValueError):
    """Raised when data violates project schema."""


MAX_OUTPUT_FILE_BYTES = 100 * 1024 * 1024


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


def log_progress(step: int, total: int, stage: str, detail: str = "") -> None:
    total_safe = max(1, total)
    pct = min(100.0, max(0.0, step / total_safe * 100.0))
    suffix = f" | {detail}" if detail else ""
    print(f"[進度] {step}/{total_safe} ({pct:.1f}%) {stage}{suffix}")


def enforce_file_size(path: Path, max_bytes: int = MAX_OUTPUT_FILE_BYTES) -> None:
    if path.exists() and path.stat().st_size > max_bytes:
        raise DataContractError(f"file too large (>100MB): {path}")


def enforce_dir_file_sizes(dirs: list[Path], max_bytes: int = MAX_OUTPUT_FILE_BYTES) -> None:
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                enforce_file_size(p, max_bytes=max_bytes)


def _shard_path(path: Path, part_idx: int) -> Path:
    return path.with_name(f"{path.stem}.part{part_idx:04d}{path.suffix}")


def shard_csv_if_needed(path: Path, max_bytes: int = MAX_OUTPUT_FILE_BYTES) -> list[Path]:
    if not path.exists():
        return []
    if path.stat().st_size <= max_bytes:
        return [path]
    df = pd.read_csv(path)
    if df.empty:
        return [path]
    part_count = max(2, int(path.stat().st_size / max_bytes) + 1)
    rows_per_part = max(1, len(df) // part_count)
    parts: list[Path] = []
    start = 0
    part_idx = 1
    while start < len(df):
        chunk = df.iloc[start : start + rows_per_part]
        p = _shard_path(path, part_idx)
        chunk.to_csv(p, index=False)
        enforce_file_size(p, max_bytes=max_bytes)
        parts.append(p)
        part_idx += 1
        start += rows_per_part
    path.unlink()
    return parts


def read_csv_maybe_sharded(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    pattern = f"{path.stem}.part*{path.suffix}"
    parts = sorted(path.parent.glob(pattern))
    if not parts:
        raise DataContractError(f"csv not found: {path}")
    frames = [pd.read_csv(p) for p in parts]
    return pd.concat(frames, ignore_index=True)
