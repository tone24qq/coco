from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from src.fetch_winwin import fetch_latest
from src.utils import DataContractError, DrawRecord


def _row_map(rows: list[DrawRecord]) -> dict[str, DrawRecord]:
    return {r.issue: r for r in rows}


def _variant_key(row: DrawRecord) -> tuple[str, tuple[int, ...], int]:
    return (row.draw_date.isoformat(), tuple(row.numbers), int(row.day_issue_index))


def _build_majority_merge(successful: dict[str, list[DrawRecord]]) -> tuple[list[DrawRecord], int]:
    by_issue: dict[str, list[DrawRecord]] = defaultdict(list)
    for rows in successful.values():
        for row in rows:
            by_issue[row.issue].append(row)

    merged: list[DrawRecord] = []
    unresolved = 0
    for issue in sorted(by_issue.keys()):
        variants: dict[tuple[str, tuple[int, ...], int], list[DrawRecord]] = defaultdict(list)
        for row in by_issue[issue]:
            variants[_variant_key(row)].append(row)
        ranked = sorted(variants.values(), key=lambda rows: len(rows), reverse=True)
        if len(ranked) > 1 and len(ranked[0]) == len(ranked[1]):
            unresolved += 1
            continue
        merged.append(ranked[0][0])
    return merged, unresolved


def run_source_consensus(
    sources: list[str],
    report_path: Path = Path("reports/source_consensus_report.json"),
    mismatch_policy: Literal["fail_fast", "majority_merge"] = "fail_fast",
) -> tuple[list[DrawRecord], dict]:
    successful: dict[str, list[DrawRecord]] = {}
    failover_reasons: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    attempts: dict[str, int] = {}

    for src in sources:
        try:
            result = fetch_latest([src])
            successful[src] = result.records
            attempts[src] = result.attempts
            failover_reasons[src] = result.failover_reason
        except Exception as exc:  # noqa: BLE001
            errors[src] = str(exc)

    if not successful:
        report = {
            "checked_sources": sources,
            "compared_issue_count": 0,
            "matched_issue_count": 0,
            "mismatched_issue_count": 0,
            "missing_issue_count_by_source": {},
            "mismatch_examples": [],
            "consensus_status": "all_failed",
            "errors": errors,
            "successful_sources": [],
            "actual_source_used": None,
            "fetch_attempts": sum(attempts.values()) if attempts else 0,
            "mismatch_policy": mismatch_policy,
            "merge_strategy": "none",
            "unresolved_mismatch_count": 0,
            "failover_reason": "all_sources_failed",
            "fetch_attempts_by_source": attempts,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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

    merged_rows, unresolved = _build_majority_merge(successful)
    if mismatched > 0 or unresolved > 0:
        status = "mismatch"
    elif any(v > 0 for v in missing_by_source.values()):
        status = "partial"

    if len(successful) == 1:
        chosen_source = list(successful.keys())[0]
    elif mismatch_policy == "fail_fast" and status == "mismatch":
        chosen_source = None
    else:
        chosen_source = "consensus_majority_merge"

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
        "fetch_attempts_by_source": attempts,
        "mismatch_policy": mismatch_policy,
        "merge_strategy": "majority_merge",
        "unresolved_mismatch_count": unresolved,
        "failover_reason": next((x for x in failover_reasons.values() if x), None),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if mismatch_policy == "fail_fast" and status == "mismatch":
        raise DataContractError("source consensus mismatch under fail_fast policy")
    return merged_rows, report
