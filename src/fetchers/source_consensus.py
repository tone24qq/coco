from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

from src.utils import REPORTS_DIR


def compare_draw_records(records_by_source: dict[str, list[dict]]) -> dict:
    by_issue: dict[int, dict[str, dict]] = defaultdict(dict)
    for source, records in records_by_source.items():
        for row in records:
            by_issue[int(row["issue"])][source] = row

    mismatches: list[dict] = []
    for issue, source_rows in sorted(by_issue.items()):
        normalized = {
            src: json.dumps(sorted(r.get("numbers", [])), ensure_ascii=False)
            for src, r in source_rows.items()
        }
        if len(set(normalized.values())) > 1:
            mismatches.append(
                {
                    "issue": issue,
                    "sources": list(source_rows.keys()),
                    "numbers_by_source": {
                        src: sorted(r.get("numbers", []))
                        for src, r in source_rows.items()
                    },
                }
            )
    return {"issue_count": len(by_issue), "mismatches": mismatches}


def detect_missing_issues(issues: Iterable[int]) -> list[int]:
    seq = sorted(set(int(x) for x in issues))
    if not seq:
        return []
    expected = set(range(seq[0], seq[-1] + 1))
    return sorted(expected - set(seq))


def detect_source_conflict(records_by_source: dict[str, list[dict]]) -> list[dict]:
    return compare_draw_records(records_by_source).get("mismatches", [])


def build_fetch_health_report(records_by_source: dict[str, list[dict]]) -> dict:
    comparison = compare_draw_records(records_by_source)
    all_issues = []
    for records in records_by_source.values():
        all_issues.extend(int(r["issue"]) for r in records)
    missing_issues = detect_missing_issues(all_issues)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            k: {
                "rows": len(v),
                "latest_issue": max([int(r["issue"]) for r in v], default=None),
            }
            for k, v in records_by_source.items()
        },
        "missing_issues": missing_issues,
        "source_conflicts": comparison.get("mismatches", []),
        "source_consensus_status": (
            "ok" if not comparison.get("mismatches") else "mismatch"
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "source_consensus_report.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "fetch_health_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
