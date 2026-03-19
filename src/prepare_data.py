from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils import DataContractError, DrawRecord, ensure_numbers, parse_date, write_processed


def _parse_numbers_row(row: dict[str, str]) -> tuple[int, ...]:
    if row.get("numbers"):
        raw = row["numbers"].strip()
        if raw.startswith("["):
            return ensure_numbers(json.loads(raw))
    nums: list[int] = []
    for key, value in row.items():
        key_norm = key.strip().lower()
        if (key_norm.startswith("n") or key.startswith("獎號")) and value.strip().isdigit():
            nums.append(int(value))
    if len(nums) == 20:
        return ensure_numbers(nums)
    raise DataContractError("unable to parse draw numbers from CSV row")


def load_history_csv(path: Path) -> list[DrawRecord]:
    records: list[DrawRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise DataContractError(f"{path} has no header")
        for row in reader:
            issue = (row.get("issue") or row.get("期數") or row.get("期別") or "").strip()
            draw_date_raw = (row.get("draw_date") or row.get("日期") or row.get("開獎日期") or "").strip()
            if not issue or not draw_date_raw:
                continue
            records.append(
                DrawRecord(
                    issue=issue,
                    draw_date=parse_date(draw_date_raw),
                    numbers=_parse_numbers_row(row),
                    day_issue_index=0,
                )
            )
    return records


def assign_day_issue_index(records: list[DrawRecord]) -> list[DrawRecord]:
    sorted_rows = sorted(records, key=lambda r: (r.draw_date, r.issue))
    counters: dict[str, int] = {}
    output: list[DrawRecord] = []
    for row in sorted_rows:
        key = row.draw_date.isoformat()
        counters[key] = counters.get(key, 0) + 1
        output.append(
            DrawRecord(
                issue=row.issue,
                draw_date=row.draw_date,
                numbers=row.numbers,
                day_issue_index=counters[key],
            )
        )
    return output


def merge_histories(paths: list[Path]) -> list[DrawRecord]:
    merged: dict[str, DrawRecord] = {}
    for path in paths:
        for row in load_history_csv(path):
            merged[row.issue] = row
    return assign_day_issue_index(list(merged.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="data/processed/history_processed.csv")
    args = parser.parse_args()

    rows = merge_histories([Path(p) for p in args.inputs])
    write_processed(Path(args.output), rows)


if __name__ == "__main__":
    main()
