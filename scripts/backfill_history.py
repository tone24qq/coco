from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json

from src.io.canonical_dataset import build_canonical_dataset
from src.io.raw_resolver import load_or_build_manifest
from src.utils import REPORTS_DIR


def main() -> None:
    df, audit = build_canonical_dataset()
    manifest = load_or_build_manifest()
    success_years = []
    fail_years = []
    for y in range(2008, 2023):
        if y in manifest.get("missing_years", []):
            fail_years.append(y)
        else:
            success_years.append(y)

    report = {
        "status": "local_first_completed",
        "canonical_rows": int(len(df)),
        "success_years": success_years,
        "failed_years": fail_years,
        "missing_issues": audit.get("missing_issues", []),
        "overlap_rate": 1.0
        - (float(audit.get("duplicate_issue_count", 0)) / max(float(len(df)), 1.0)),
        "official_repair_required": bool(
            fail_years or audit.get("missing_issue_count", 0) > 0
        ),
        "repair_priority": [
            "official result_download",
            "official history_result",
            "official current result/schedule",
            "winwin",
            "other mirrors",
        ],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "history_backfill_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # lightweight ablation placeholders from local-first policy
    table_path = REPORTS_DIR / "history_ablation_table.csv"
    table_path.write_text(
        "variant,description,status\n"
        "A,2023-2026 only,ready\n"
        "B,2023-2026 + long_history priors,ready\n"
        "C,2008-2026 full equal-weight,ready\n"
        "D,recent training + long_history decay priors,ready\n",
        encoding="utf-8",
    )
    summary = {
        "status": "prepared",
        "note": "Local-first dataset is ready. Run training/backtest pipeline to fill performance metrics.",
    }
    (REPORTS_DIR / "history_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
