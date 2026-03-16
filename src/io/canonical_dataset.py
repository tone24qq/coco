from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.io.raw_resolver import (
    DEFAULT_RAW_DIR,
    build_raw_manifest,
    read_raw_csv_to_standard_df,
    resolve_raw_csv_paths,
)
from src.utils import DATA_PROCESSED_DIR, REPORTS_DIR

CANONICAL_CSV = DATA_PROCESSED_DIR / "bingo_draws_canonical.csv"
CANONICAL_PARQUET = DATA_PROCESSED_DIR / "bingo_draws_canonical.parquet"
AUDIT_JSON = REPORTS_DIR / "local_data_audit.json"


def _raw_hash(issue: int, draw_date: str, numbers: str) -> str:
    payload = f"{issue}|{draw_date}|{numbers}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_canonical_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[pd.DataFrame, dict]:
    manifest = build_raw_manifest(raw_dir=raw_dir)
    files = resolve_raw_csv_paths(raw_dir=raw_dir)
    ok_files: list[str] = []
    failed_files: list[dict] = []
    frames: list[pd.DataFrame] = []
    column_summaries: dict[str, list[str]] = {}

    for path in files:
        try:
            df = read_raw_csv_to_standard_df(path)
            column_summaries[path.name] = list(df.columns)
            frames.append(df)
            ok_files.append(path.name)
        except Exception as exc:  # noqa: BLE001
            failed_files.append({"file": path.name, "error": str(exc)})

    if not frames:
        raise ValueError("no readable local CSVs for canonical dataset")

    full = pd.concat(frames, ignore_index=True)
    full["raw_hash"] = full.apply(
        lambda r: _raw_hash(int(r["issue"]), str(r["draw_date"]), str(r["numbers"])),
        axis=1,
    )
    dup_issues = (
        full[full.duplicated(subset=["issue"], keep=False)]["issue"]
        .astype(int)
        .tolist()
    )

    full = full.drop_duplicates(subset=["issue"], keep="first").sort_values("issue")
    full = full.reset_index(drop=True)

    issues = full["issue"].astype(int).tolist()
    missing_issues: list[int] = []
    if issues:
        min_issue = min(issues)
        max_issue = max(issues)
        # Bingo issue ids may be non-dense across historical exports; avoid exploding gaps.
        if (max_issue - min_issue) <= int(len(issues) * 3):
            seq = set(range(min_issue, max_issue + 1))
            missing_issues = sorted(seq - set(issues))

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(CANONICAL_CSV, index=False)
    full.to_parquet(CANONICAL_PARQUET, index=False)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_file_count": int(manifest.get("file_count", 0)),
        "detected_files": [e["original_filename"] for e in manifest.get("entries", [])],
        "loaded_files": ok_files,
        "failed_files": failed_files,
        "coverage_year_start": manifest.get("coverage_year_start"),
        "coverage_year_end": manifest.get("coverage_year_end"),
        "missing_years": manifest.get("missing_years", []),
        "canonical_rows": int(len(full)),
        "issue_start": int(min(issues)) if issues else None,
        "issue_end": int(max(issues)) if issues else None,
        "missing_issues": missing_issues[:5000],
        "missing_issue_count": int(len(missing_issues)),
        "duplicate_issues": sorted(set(int(x) for x in dup_issues))[:5000],
        "duplicate_issue_count": int(len(set(dup_issues))),
        "column_summary": column_summaries,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return full, audit


def load_canonical_or_build() -> pd.DataFrame:
    if CANONICAL_PARQUET.exists():
        return pd.read_parquet(CANONICAL_PARQUET)
    if CANONICAL_CSV.exists():
        return pd.read_csv(CANONICAL_CSV)
    raise FileNotFoundError(
        "canonical dataset not found; run `python src/prepare_data.py` first"
    )


def load_canonical_with_diagnostics() -> tuple[pd.DataFrame, dict]:
    start = time.perf_counter()
    if CANONICAL_PARQUET.exists():
        df = pd.read_parquet(CANONICAL_PARQUET)
        source = str(CANONICAL_PARQUET)
    elif CANONICAL_CSV.exists():
        df = pd.read_csv(CANONICAL_CSV)
        source = str(CANONICAL_CSV)
    else:
        raise FileNotFoundError(
            "canonical dataset not found; run `python src/prepare_data.py` first"
        )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return df, {
        "source": source,
        "rows": int(len(df)),
        "elapsed_ms": elapsed_ms,
    }
