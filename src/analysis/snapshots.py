from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.utils import DrawRecord


def _window_summary(records: list[DrawRecord], window: int) -> dict:
    rows = records[-window:] if window <= len(records) else records
    freq: Counter[int] = Counter()
    for r in rows:
        freq.update(r.numbers)
    return {
        "window": window,
        "rows": len(rows),
        "top_numbers": [[int(n), int(c)] for n, c in freq.most_common(10)],
    }


def build_history_snapshot(records: list[DrawRecord], output_path: Path = Path("reports/history_snapshot.json")) -> dict:
    if not records:
        snapshot = {
            "total_history_rows": 0,
            "issue_range": [None, None],
            "date_range": [None, None],
            "coverage_year_start": None,
            "coverage_year_end": None,
            "per_year_row_counts": {},
            "per_number_total_hits": {},
            "recent_window_summaries": {},
        }
    else:
        issues = [r.issue for r in records]
        dates = [r.draw_date for r in records]
        per_year = Counter(d.year for d in dates)
        num_counter: Counter[int] = Counter()
        for r in records:
            num_counter.update(r.numbers)
        snapshot = {
            "total_history_rows": len(records),
            "issue_range": [min(issues), max(issues)],
            "date_range": [min(dates).isoformat(), max(dates).isoformat()],
            "coverage_year_start": int(min(per_year.keys())),
            "coverage_year_end": int(max(per_year.keys())),
            "per_year_row_counts": {str(k): int(v) for k, v in sorted(per_year.items())},
            "per_number_total_hits": {str(k): int(v) for k, v in sorted(num_counter.items())},
            "recent_window_summaries": {str(w): _window_summary(records, w) for w in [20, 50, 100, 200]},
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot


def read_history_snapshot(path: Path = Path("reports/history_snapshot.json")) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
