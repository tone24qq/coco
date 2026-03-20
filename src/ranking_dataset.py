from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils import DataContractError, enforce_dir_file_sizes, enforce_file_size, log_progress, read_csv_maybe_sharded, shard_csv_if_needed


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
    frame = read_csv_maybe_sharded(path)
    rows = frame.to_dict(orient="records")
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
    shard_csv_if_needed(path)
    if path.exists():
        enforce_file_size(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feature_store/ranking_features.csv")
    parser.add_argument("--output", default="data/feature_store/ranking_dataset.csv")
    args = parser.parse_args()

    log_progress(1, 4, "讀取 feature store", f"輸入={args.input}")
    rows = load_feature_rows(Path(args.input))
    log_progress(2, 4, "驗證 80 候選合約", f"rows={len(rows)}")
    validate_group_contract(rows)
    log_progress(3, 4, "附加 group_id", "開始")
    with_group = attach_group_ids(rows)
    write_rows(Path(args.output), with_group)
    enforce_dir_file_sizes([Path("data/feature_store"), Path("reports"), Path("models")])
    log_progress(4, 4, "ranking_dataset 輸出完成", f"輸出={args.output}")


if __name__ == "__main__":
    main()
