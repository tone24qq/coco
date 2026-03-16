from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.analysis.snapshots import build_history_snapshot
from src.fetchers.auzo_bingo import (
    BingoDrawFetcher,
    FetchDrawsError,
    build_recent_draws,
)
from src.io.artifact_guard import write_parquet_with_size_guard
from src.io.canonical_dataset import CANONICAL_PARQUET, load_canonical_or_build
from src.utils import CONFIG_DIR, load_yaml


def _persist_canonical(df: pd.DataFrame, artifact_mode: str) -> None:
    write_parquet_with_size_guard(
        df.sort_values("issue").reset_index(drop=True),
        output_path=CANONICAL_PARQUET,
        artifact_mode=artifact_mode,
        preferred_codec="zstd",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update canonical and snapshot incrementally"
    )
    parser.add_argument(
        "--artifact-mode",
        choices=["runtime", "export"],
        default="runtime",
        help="runtime keeps single parquet; export may shard when size guard triggers",
    )
    args = parser.parse_args()

    cfg = load_yaml(CONFIG_DIR / "predict.yaml")
    sources = cfg.get("auto_fetch_sources", [])
    timeout = float(cfg.get("fetch_timeout_seconds", 8.0))
    retries = int(cfg.get("fetch_retries", 2))
    backoff = float(cfg.get("fetch_retry_backoff_seconds", 0.5))

    canonical = (
        load_canonical_or_build(artifact_mode=args.artifact_mode)
        .sort_values("issue")
        .reset_index(drop=True)
    )
    latest_before = int(canonical["issue"].max()) if not canonical.empty else None

    fetcher = BingoDrawFetcher(
        sources=sources,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=backoff,
    )

    appended = 0
    fetch_status = "ok"
    fetch_error = None
    fetched_range = None
    try:
        _, fetched_records, source, _ = build_recent_draws(
            fetcher=fetcher,
            min_draws=1,
            max_draws=3000,
        )
        incoming = pd.DataFrame(
            {
                "issue": [int(r.issue) for r in fetched_records],
                "draw_date": [str(r.draw_time or "") for r in fetched_records],
                "numbers": [
                    json.dumps(sorted(r.numbers), ensure_ascii=False)
                    for r in fetched_records
                ],
                "numbers_draw_order": [
                    json.dumps(r.numbers, ensure_ascii=False) for r in fetched_records
                ],
                "draw_time": [r.draw_time for r in fetched_records],
                "consecutive_count": [r.streak_count for r in fetched_records],
                "size": [r.size_label for r in fetched_records],
                "odd_even": [r.odd_even_label for r in fetched_records],
                "source": [source for _ in fetched_records],
                "source_priority": [4 for _ in fetched_records],
                "raw_file": ["live_fetch" for _ in fetched_records],
                "raw_hash": ["" for _ in fetched_records],
            }
        )
        if latest_before is not None:
            incoming = incoming[incoming["issue"] > latest_before].copy()
        appended = int(len(incoming))
        if appended > 0:
            canonical = (
                pd.concat([canonical, incoming], ignore_index=True)
                .drop_duplicates(subset=["issue"], keep="first")
                .sort_values("issue")
                .reset_index(drop=True)
            )
            _persist_canonical(canonical, artifact_mode=args.artifact_mode)
            fetched_range = [int(incoming["issue"].min()), int(incoming["issue"].max())]
    except FetchDrawsError as exc:
        fetch_status = "degraded"
        fetch_error = str(exc)

    snapshot, meta = build_history_snapshot(canonical, artifact_mode=args.artifact_mode)
    latest_after = int(canonical["issue"].max()) if not canonical.empty else None

    report = {
        "status": "ok",
        "artifact_mode": args.artifact_mode,
        "fetch_status": fetch_status,
        "fetch_error": fetch_error,
        "canonical_issue_before": latest_before,
        "canonical_issue_after": latest_after,
        "incremental_issues_added": appended,
        "incremental_issue_range": fetched_range,
        "snapshot_rows": int(len(snapshot)),
        "snapshot_type_counts": meta.get("snapshot_type_counts", {}),
        "snapshot_path": meta.get("paths", {}).get("history_snapshot"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
