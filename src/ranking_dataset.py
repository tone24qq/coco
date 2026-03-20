from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils import DataContractError


def attach_group_ids(feature_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, int] = {}
    output: list[dict[str, str]] = []
    for row in feature_rows:
        issue = row["issue"]
        if issue not in groups:
            groups[issue] = len(groups)
        row_copy = dict(row)
        row_copy["group_id"] = str(groups[issue])
        output.append(row_copy)
    return output


def load_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise DataContractError("feature store is empty")
    return rows


def validate_group_contract(rows: list[dict[str, str]]) -> None:
    by_issue: dict[str, int] = {}
    for row in rows:
        by_issue[row["issue"]] = by_issue.get(row["issue"], 0) + 1
    wrong = [issue for issue, size in by_issue.items() if size != 80]
    if wrong:
        raise DataContractError(f"each issue must have 80 candidates: {wrong[:3]}")


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feature_store/ranking_features.csv")
    parser.add_argument("--output", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    rows = load_feature_rows(Path(args.input))
    validate_group_contract(rows)
    with_group = attach_group_ids(rows)
    write_rows(Path(args.output), with_group)


if __name__ == "__main__":
    main()
