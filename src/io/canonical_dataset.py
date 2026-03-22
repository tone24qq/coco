from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.io.raw_resolver import build_raw_manifest
from src.prepare_data import load_history_csv, merge_histories
from src.utils import DataContractError, DrawRecord


def _validate_header(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = [f.strip() for f in (reader.fieldnames or [])]
    issue_ok = any(x in fields for x in ["issue", "期別", "期數"])
    date_ok = any(x in fields for x in ["draw_date", "開獎日期", "日期"])
    num_ok = any(x.startswith("獎號") for x in fields) or "numbers" in fields
    if not (issue_ok and date_ok and num_ok):
        raise DataContractError(f"invalid header in raw file: {path}")


def _summary(raw_records: list[DrawRecord], canonical_records: list[DrawRecord], detected_files: list[str]) -> dict:
    base_rows = raw_records if raw_records else canonical_records
    if not base_rows:
        return {
            "detected_files": detected_files,
            "file_count": len(detected_files),
            "issue_range": [None, None],
            "date_range": [None, None],
            "total_rows": 0,
            "duplicate_issue_count": 0,
            "duplicate_issue_examples": [],
            "missing_issue_count": "estimated: unavailable",
            "coverage_year_start": None,
            "coverage_year_end": None,
            "per_year_row_counts": {},
            "canonical_rows": 0,
        }

    issues = [r.issue for r in base_rows]
    dates = [r.draw_date for r in base_rows]
    dup_counter = Counter(issues)
    dup_examples = [k for k, v in dup_counter.items() if v > 1][:10]
    per_year = Counter(d.year for d in dates)

    missing = "estimated: unavailable"
    unique_issues = sorted(set(issues))
    if unique_issues and all(i.isdigit() for i in unique_issues):
        values = sorted(int(i) for i in unique_issues)
        expected = values[-1] - values[0] + 1
        missing = max(0, expected - len(values))

    return {
        "detected_files": detected_files,
        "file_count": len(detected_files),
        "issue_range": [min(issues), max(issues)],
        "date_range": [min(dates).isoformat(), max(dates).isoformat()],
        "total_rows": len(base_rows),
        "duplicate_issue_count": int(sum(v - 1 for v in dup_counter.values() if v > 1)),
        "duplicate_issue_examples": dup_examples,
        "missing_issue_count": missing,
        "coverage_year_start": int(min(per_year.keys())),
        "coverage_year_end": int(max(per_year.keys())),
        "per_year_row_counts": {str(k): int(v) for k, v in sorted(per_year.items())},
        "canonical_rows": len(canonical_records),
    }


def build_canonical_audit(
    raw_dirs: list[Path] | None = None,
    audit_output_path: Path = Path("reports/local_data_audit.json"),
    manifest_output_path: Path = Path("reports/raw_manifest.json"),
) -> tuple[dict, list[DrawRecord]]:
    manifest = build_raw_manifest(raw_dirs=raw_dirs, output_path=manifest_output_path)
    paths = [Path(p) for p in manifest["detected_files"]]

    raw_records: list[DrawRecord] = []
    for p in paths:
        _validate_header(p)
        rows = load_history_csv(p)
        raw_records.extend(rows)

    canonical_records = merge_histories(paths) if paths else []
    audit = _summary(raw_records, canonical_records, manifest["detected_files"])
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit, canonical_records


def read_audit_summary(path: Path = Path("reports/local_data_audit.json")) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
