from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.features import parse_numbers_column
from src.io.artifact_guard import write_parquet_with_size_guard
from src.io.canonical_dataset import load_canonical_or_build
from src.utils import DATA_PROCESSED_DIR

SNAPSHOT_PARQUET = DATA_PROCESSED_DIR / "history_snapshot.parquet"
SNAPSHOT_META = DATA_PROCESSED_DIR / "history_snapshot_meta.json"
SNAPSHOT_CSV = DATA_PROCESSED_DIR / "history_snapshot.csv"

_SNAPSHOT_CACHE: dict | None = None
_SNAPSHOT_CACHE_KEY: tuple[float, float] | None = None


def _normalize_draw_day(df: pd.DataFrame) -> pd.Series:
    date_col = df.get("draw_date")
    if date_col is None:
        return pd.Series(["unknown"] * len(df))
    out = date_col.astype(str).str.slice(0, 10)
    return out.str.replace("/", "-", regex=False)


def _window_stats(values: pd.Series, windows: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    arr = values.astype(float)
    for w in windows:
        tail = arr.iloc[-min(w, len(arr)) :]
        out[f"mean_{w}"] = float(tail.mean()) if len(tail) else 0.0
        out[f"std_{w}"] = float(tail.std(ddof=0)) if len(tail) else 0.0
    return out


def _build_board_priors(issue_stats: pd.DataFrame) -> dict:
    if issue_stats.empty:
        return {}
    windows = [200, 500, 1000]
    zone_keys = ["zone_A", "zone_B", "zone_C", "zone_D"]
    board = {
        "odd_even": _window_stats(issue_stats["odd_count"], windows),
        "big_small": _window_stats(issue_stats["big_count"], windows),
        "sum": _window_stats(issue_stats["issue_sum"], windows),
        "average": _window_stats(issue_stats["issue_average"], windows),
        "span": _window_stats(issue_stats["issue_span"], windows),
        "min": _window_stats(issue_stats["issue_min"], windows),
        "max": _window_stats(issue_stats["issue_max"], windows),
        "zones": {},
    }
    for z in zone_keys:
        board["zones"][z] = _window_stats(issue_stats[z], windows)

    tail = issue_stats.tail(min(80, len(issue_stats)))
    board["day_pace"] = {
        "avg_issue_gap_recent": (
            float(tail["issue"].astype(float).diff().dropna().mean())
            if len(tail) >= 2
            else 1.0
        ),
        "sum_switch_ratio_recent": float(
            (tail["issue_sum"].diff().fillna(0) != 0).astype(float).mean()
        ),
        "span_switch_ratio_recent": float(
            (tail["issue_span"].diff().fillna(0) != 0).astype(float).mean()
        ),
    }
    return board


def _build_number_stats(
    draws: list[list[int]],
    draw_days: list[str],
) -> pd.DataFrame:
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
            "current_gap_all": int(
                (total - 1 - hit_indices[-1]) if hit_indices else total
            ),
            "avg_gap_all": float(np.mean(gaps)) if gaps else float(total),
            "max_gap_all": int(max(gaps)) if gaps else int(total),
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
        rows.append(
            {
                "snapshot_type": "day",
                "entity_key": str(day),
                "draw_day": str(day),
                "day_hot_rank": json.dumps(
                    {
                        "odd_count": int(day_df["odd_count"].mean()),
                        "big_count": int(day_df["big_count"].mean()),
                    },
                    ensure_ascii=False,
                ),
                "day_cold_rank": json.dumps(
                    {
                        "even_count": int(day_df["even_count"].mean()),
                        "small_count": int(day_df["small_count"].mean()),
                    },
                    ensure_ascii=False,
                ),
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
    artifact_mode: str = "runtime",
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
    board_priors = _build_board_priors(issue_df)

    snapshot = pd.concat([num_df, issue_df, day_df], ignore_index=True, sort=False)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    write_result, write_summary = write_parquet_with_size_guard(
        snapshot,
        output_path=SNAPSHOT_PARQUET,
        artifact_mode=artifact_mode,
        preferred_codec="zstd",
    )
    if write_result.format == "parquet":
        SNAPSHOT_CSV.unlink(missing_ok=True)

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
        "board_priors": board_priors,
        "artifact_mode": artifact_mode,
        "snapshot_format": write_result.format,
        "snapshot_compression": write_result.compression,
        "size_guard": write_summary,
        "paths": {
            "history_snapshot": str(write_result.path),
            "history_snapshot_meta": str(SNAPSHOT_META),
            "history_snapshot_manifest": write_result.manifest_path,
        },
    }
    SNAPSHOT_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot, meta


def _load_snapshot_df(meta: dict | None = None) -> pd.DataFrame:
    if SNAPSHOT_PARQUET.exists():
        return pd.read_parquet(SNAPSHOT_PARQUET)
    if SNAPSHOT_CSV.exists():
        return pd.read_csv(SNAPSHOT_CSV)

    snapshot_path = str((meta or {}).get("paths", {}).get("history_snapshot", ""))
    if snapshot_path:
        path = Path(snapshot_path)
        if path.is_dir():
            return pd.read_parquet(path)
        if path.suffix == ".parquet" and path.exists():
            return pd.read_parquet(path)
        if path.suffix == ".csv" and path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def load_history_snapshot_payload(force_reload: bool = False) -> dict:
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_KEY
    meta_mtime = SNAPSHOT_META.stat().st_mtime if SNAPSHOT_META.exists() else 0.0
    data_path = SNAPSHOT_PARQUET if SNAPSHOT_PARQUET.exists() else SNAPSHOT_CSV
    data_mtime = data_path.stat().st_mtime if data_path.exists() else 0.0
    cache_key = (meta_mtime, data_mtime)
    if (
        not force_reload
        and _SNAPSHOT_CACHE is not None
        and _SNAPSHOT_CACHE_KEY == cache_key
    ):
        return _SNAPSHOT_CACHE

    start = time.perf_counter()
    meta = (
        json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
        if SNAPSHOT_META.exists()
        else {}
    )
    df = _load_snapshot_df(meta=meta)
    number_df = (
        df[df.get("snapshot_type", "") == "number"].copy()
        if not df.empty
        else pd.DataFrame()
    )
    if not number_df.empty and "number" in number_df.columns:
        number_df["number"] = (
            pd.to_numeric(number_df["number"], errors="coerce").fillna(0).astype(int)
        )
        number_df = number_df.set_index("number", drop=False)

    payload = {
        "status": "ready" if not number_df.empty else "unavailable",
        "meta": meta,
        "number_priors": number_df,
        "load_elapsed_ms": int((time.perf_counter() - start) * 1000),
    }
    _SNAPSHOT_CACHE = payload
    _SNAPSHOT_CACHE_KEY = cache_key
    return payload
