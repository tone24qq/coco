from __future__ import annotations

from src.fetchers.source_consensus import (
    build_fetch_health_report,
    compare_draw_records,
    detect_missing_issues,
)


def test_source_consensus_and_missing() -> None:
    records = {
        "official": [
            {"issue": 1, "numbers": [1, 2, 3]},
            {"issue": 2, "numbers": [2, 3, 4]},
        ],
        "winwin": [
            {"issue": 1, "numbers": [1, 2, 3]},
            {"issue": 2, "numbers": [2, 3, 5]},
        ],
    }
    cmp = compare_draw_records(records)
    assert cmp["issue_count"] == 2
    assert len(cmp["mismatches"]) == 1

    missing = detect_missing_issues([1, 3, 4])
    assert missing == [2]

    report = build_fetch_health_report(records)
    assert report["source_consensus_status"] == "mismatch"
