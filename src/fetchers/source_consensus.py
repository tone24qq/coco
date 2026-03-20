from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.fetch_winwin import fetch_latest
from src.utils import DataContractError, DrawRecord


def _row_map(rows: list[DrawRecord]) -> dict[str, DrawRecord]:
    return {r.issue: r for r in rows}


def run_source_consensus(
    sources: list[str], report_path: Path = Path("reports/source_consensus_report.json")
) -> tuple[list[DrawRecord], dict]:
    successful: dict[str, list[DrawRecord]] = {}
    errors: dict[str, str] = {}
    attempts: dict[str, int] = {}

    for src in sources:
        try:
            result = fetch_latest([src])
            successful[src] = result.records
            attempts[src] = result.attempts
        except Exception as exc:  # noqa: BLE001
            errors[src] = str(exc)

    if not successful:
        raise DataContractError("all fetch sources failed")

    status = "single_source" if len(successful) == 1 else "ok"
    mismatch_examples: list[dict] = []
    missing_by_source: dict[str, int] = defaultdict(int)
    matched = 0
    mismatched = 0
    compared = 0

    if len(successful) > 1:
        issue_pool = set()
        maps = {k: _row_map(v) for k, v in successful.items()}
        for m in maps.values():
            issue_pool.update(m.keys())
        for issue in sorted(issue_pool):
            rows = [m.get(issue) for m in maps.values()]
            present_rows = [r for r in rows if r is not None]
            if len(present_rows) < 2:
                compared += 1
                for src, m in maps.items():
                    if issue not in m:
                        missing_by_source[src] += 1
                continue
            compared += 1
            base = present_rows[0]
            ok = all((r.draw_date == base.draw_date and set(r.numbers) == set(base.numbers)) for r in present_rows[1:])
            if ok:
                matched += 1
            else:
                mismatched += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "issue": issue,
                            "sources": {
                                src: {
                                    "draw_date": maps[src][issue].draw_date.isoformat(),
                                    "numbers": list(maps[src][issue].numbers),
                                }
                                for src in maps
                                if issue in maps[src]
                            },
                        }
                    )
        if mismatched > 0:
            status = "mismatch"
        elif any(v > 0 for v in missing_by_source.values()):
            status = "partial"

    chosen_source = sorted(successful.keys(), key=lambda s: len(successful[s]), reverse=True)[0]
    report = {
        "checked_sources": sources,
        "compared_issue_count": compared,
        "matched_issue_count": matched,
        "mismatched_issue_count": mismatched,
        "missing_issue_count_by_source": dict(missing_by_source),
        "mismatch_examples": mismatch_examples,
        "consensus_status": status,
        "errors": errors,
        "successful_sources": list(successful.keys()),
        "actual_source_used": chosen_source,
        "fetch_attempts": sum(attempts.values()) if attempts else 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return successful[chosen_source], report
