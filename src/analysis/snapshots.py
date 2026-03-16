from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.analysis.features import parse_numbers_column
from src.io.canonical_dataset import load_canonical_or_build
from src.utils import DATA_PROCESSED_DIR

SNAPSHOT_PARQUET = DATA_PROCESSED_DIR / "history_snapshot.parquet"
SNAPSHOT_META = DATA_PROCESSED_DIR / "history_snapshot_meta.json"
SNAPSHOT_CSV = DATA_PROCESSED_DIR / "history_snapshot.csv"


def _normalize_draw_day(df: pd.DataFrame) -> pd.Series:
    date_col = df.get("draw_date")
    if date_col is None:
        return pd.Series(["unknown"] * len(df))
    out = date_col.astype(str).str.slice(0, 10)
    out = out.str.replace("/", "-", regex=False)
    return out


def _build_number_stats(draws: list[list[int]], draw_days: list[str]) -> pd.DataFrame:
    total = len(draws)
    latest = set(draws[-1]) if draws else set()
    prev = set(draws[-2]) if len(draws) >= 2 else set()
    today = draw_days[-1] if draw_days else "unknown"
    today_draws = [set(draw) for draw, day in zip(draws, draw_days) if day == today]

    rows: list[dict] = []
    windows = [5, 10, 20, 50, 100, 200, 500, 1000]
    for num in range(1, 81):
        hit_indices = [idx for idx, draw in enumerate(draws) if num in draw]
        hit_count = len(hit_indices)
        gaps = [b - a for a, b in zip(hit_indices, hit_indices[1:])]
        row = {
            "snapshot_type": "number",
            "entity_key": str(num),
            "number": int(num),
            "total_hits_all_time": int(hit_count),
            "current_gap": int((total - 1 - hit_indices[-1]) if hit_indices else total),
            "avg_gap": float(np.mean(gaps)) if gaps else None,
            "max_gap": int(max(gaps)) if gaps else None,
            "today_hits": int(sum(1 for draw in today_draws if num in draw)),
            "carryover_from_prev": int(num in latest and num in prev),
            "pm1_neighbor_hits": int(sum(1 for n in latest if abs(n - num) == 1)),
            "pm2_neighbor_hits": int(sum(1 for n in latest if abs(n - num) == 2)),
            "draw_day": today,
        }
        for w in windows:
            tail = draws[-min(w, total) :]
            row[f"hits_last_{w}"] = int(sum(1 for draw in tail if num in draw))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_issue_stats(df: pd.DataFrame, draws: list[list[int]]) -> pd.DataFrame:
    rows: list[dict] = []
    day_series = _normalize_draw_day(df).tolist()
    for i, draw in enumerate(draws):
        arr = sorted(draw)
        odd = sum(1 for n in arr if n % 2 == 1)
        small = sum(1 for n in arr if n <= 40)
        rows.append(
            {
                "snapshot_type": "issue",
                "entity_key": str(int(df.iloc[i]["issue"])),
                "issue": int(df.iloc[i]["issue"]),
                "draw_day": day_series[i],
                "odd_count": int(odd),
                "even_count": int(20 - odd),
                "small_count": int(small),
                "big_count": int(20 - small),
                "zone_A": int(sum(1 for n in arr if n <= 20)),
                "zone_B": int(sum(1 for n in arr if 21 <= n <= 40)),
                "zone_C": int(sum(1 for n in arr if 41 <= n <= 60)),
                "zone_D": int(sum(1 for n in arr if 61 <= n <= 80)),
                "issue_sum": int(sum(arr)),
                "issue_average": float(np.mean(arr)),
                "issue_span": int(max(arr) - min(arr)),
                "issue_min": int(min(arr)),
                "issue_max": int(max(arr)),
            }
        )
    return pd.DataFrame(rows)


def _build_day_stats(issue_stats: pd.DataFrame) -> pd.DataFrame:
    if issue_stats.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    issue_stats = issue_stats.sort_values("issue")
    for day, day_df in issue_stats.groupby("draw_day"):
        day_issues = day_df["issue"].astype(int).tolist()
        for issue in day_issues:
            target = issue_stats[issue_stats["issue"] == issue].iloc[0]
            # reconstruct approximation from aggregate fields is not feasible; use simple RI-like counts from zones.
            # keep stable schema by deriving from issue parity/size dynamics.
            _ = target
        # RI-like proxy from issue-level odd/even and big/small counts.
        ri_like = {
            "draw_count": int(len(day_df)),
            "avg_odd_count": float(day_df["odd_count"].mean()),
            "avg_big_count": float(day_df["big_count"].mean()),
        }
        rj_like = {
            "sum_switch_count": int(
                (day_df["issue_sum"].diff().fillna(0) != 0).astype(int).sum()
            ),
            "span_switch_count": int(
                (day_df["issue_span"].diff().fillna(0) != 0).astype(int).sum()
            ),
            "avg_issue_gap": (
                float(np.mean(np.diff(day_issues))) if len(day_issues) >= 2 else None
            ),
        }

        hot_rank = {
            "odd_count": int(day_df["odd_count"].mean()),
            "big_count": int(day_df["big_count"].mean()),
        }
        cold_rank = {
            "even_count": int(day_df["even_count"].mean()),
            "small_count": int(day_df["small_count"].mean()),
        }

        rows.append(
            {
                "snapshot_type": "day",
                "entity_key": str(day),
                "draw_day": str(day),
                "day_hot_rank": json.dumps(hot_rank, ensure_ascii=False),
                "day_cold_rank": json.dumps(cold_rank, ensure_ascii=False),
                "ri_like_counts": json.dumps(ri_like, ensure_ascii=False),
                "rj_like_drag_skip": json.dumps(rj_like, ensure_ascii=False),
                "issue_count": int(len(day_df)),
                "issue_start": int(min(day_issues)),
                "issue_end": int(max(day_issues)),
            }
        )
    return pd.DataFrame(rows)


def build_history_snapshot(
    canonical_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    if canonical_df is None:
        canonical_df = load_canonical_or_build()

    df = canonical_df.copy().sort_values("issue").reset_index(drop=True)
    if "numbers" not in df.columns:
        raise ValueError("canonical dataset missing numbers column")

    draws = [parse_numbers_column(v) for v in df["numbers"].astype(str).tolist()]
    draw_days = _normalize_draw_day(df).tolist()

    num_df = _build_number_stats(draws, draw_days)
    issue_df = _build_issue_stats(df, draws)
    day_df = _build_day_stats(issue_df)

    snapshot = pd.concat([num_df, issue_df, day_df], ignore_index=True, sort=False)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_format = "parquet"
    try:
        snapshot.to_parquet(SNAPSHOT_PARQUET, index=False)
    except (ImportError, ValueError, OSError):
        snapshot.to_csv(SNAPSHOT_CSV, index=False)
        snapshot_format = "csv"

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_rows": int(len(df)),
        "canonical_issue_start": int(df["issue"].min()) if not df.empty else None,
        "canonical_issue_end": int(df["issue"].max()) if not df.empty else None,
        "snapshot_rows": int(len(snapshot)),
        "snapshot_type_counts": {
            "number": int(len(num_df)),
            "issue": int(len(issue_df)),
            "day": int(len(day_df)),
        },
        "snapshot_format": snapshot_format,
        "paths": {
            "history_snapshot": str(
                SNAPSHOT_PARQUET if snapshot_format == "parquet" else SNAPSHOT_CSV
            ),
            "history_snapshot_meta": str(SNAPSHOT_META),
        },
    }
    SNAPSHOT_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot, meta
