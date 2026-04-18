from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.masking_dataset import (
    MaskingConfig,
    iter_masked_ranking_dataset_chunks,
)
from src.safe_io import SafeWriteConfig, write_dataframe_chunks_safe
from src.whole_board_features import (
    DEPRECATED_FEATURE_PREFIXES,
    FEATURE_MERGE_MAP,
    FEATURE_RENAME_MAP,
    FEATURE_SCHEMA_VERSION,
    NEAR_CONSTANT_DOMINANT_RATIO,
    NEAR_CONSTANT_STD_EPS,
    is_primary_feature_column,
)


def _enable_fault_handler() -> None:
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        faulthandler.enable()
    except Exception:
        pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _feature_category(col: str) -> str:
    name = col.replace("board_state_", "").replace("candidate_delta_", "")
    if name.startswith("global_") or name.startswith("residue_target_") or name.startswith("multiple10_target_"):
        return "target_global_features"
    if name.startswith("local5x5_"):
        return "local_window_features"
    if name.startswith("row_") or name.startswith("col_"):
        return "row_col_features"
    if name.startswith("neighbor_"):
        return "neighbor_features"
    if "interaction" in name:
        return "interaction_features"
    if "hist" in name or "mode_bin" in name:
        return "histogram_features"
    if any(name.startswith(pref) for pref in DEPRECATED_FEATURE_PREFIXES):
        return "deprecated_features"
    return "kept_features"


def _merge_value_counts(dst: Dict[float, int], src: pd.Series) -> None:
    for value, cnt in src.items():
        dst[float(value)] = int(dst.get(float(value), 0) + int(cnt))


def _update_size_agg(size_agg: Dict[str, Dict[str, Dict[str, float]]], size: str, col: str, s: pd.Series) -> None:
    cell = size_agg[size].setdefault(col, {"n": 0.0, "sum": 0.0, "sum_sq": 0.0, "nunique": 0.0})
    values = s.fillna(0.0).astype(float)
    n = float(len(values))
    cell["n"] += n
    cell["sum"] += float(values.sum())
    cell["sum_sq"] += float((values * values).sum())
    cell["nunique"] += float(values.nunique(dropna=False) > 1)


def _std_from_agg(n: float, s: float, s2: float) -> float:
    if n <= 1:
        return 0.0
    mean = s / n
    var = max(s2 / n - mean * mean, 0.0)
    return float(var**0.5)


def _peak_rss_mb() -> float | None:
    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return None


def main() -> None:
    _enable_fault_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/full_boards/full_board_corpus.jsonl")
    parser.add_argument("--synthetic-corpus", default="data/full_boards/synthetic_board_corpus.jsonl")
    parser.add_argument("--output", default="data/ranking/ranking_dataset.parquet")
    parser.add_argument("--mask-ratios", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--masks-per-ratio", type=int, default=2)
    parser.add_argument("--shard-rows", type=int, default=150000)
    parser.add_argument("--feature-schema", default="artifacts/feature_schema_residue.json")
    parser.add_argument("--max-file-mb", type=int, default=100)
    parser.add_argument("--feature-audit-before", default="reports/feature_audit_before.json")
    parser.add_argument("--feature-audit-after", default="reports/feature_audit_after.json")
    parser.add_argument("--feature-merge-map", default="reports/feature_merge_map.json")
    parser.add_argument("--dead-feature-report", default="reports/dead_feature_report.json")
    parser.add_argument("--size-aware-summary", default="reports/size_aware_feature_summary.json")
    parser.add_argument("--debug-crash-report", action="store_true")
    args = parser.parse_args()

    real_rows = read_jsonl(Path(args.real_corpus))
    synth_rows: List[Dict[str, Any]] = []
    synth_path = Path(args.synthetic_corpus)
    if synth_path.exists():
        synth_rows = read_jsonl(synth_path)

    boards = real_rows + synth_rows
    ratios = [float(x.strip()) for x in args.mask_ratios.split(",") if x.strip()]
    config = MaskingConfig(ratios=ratios, masks_per_ratio=args.masks_per_ratio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.is_file():
        output_path.unlink()
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)

    all_columns: List[str] = []
    primary_cols: List[str] = []
    row_count = 0
    group_ids: set[str] = set()
    size_rows: Dict[str, int] = defaultdict(int)
    size_groups: Dict[str, set[str]] = defaultdict(set)
    stats: Dict[str, Dict[str, Any]] = {}
    size_agg: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    crash_ctx: Dict[str, Any] = {"stage": "stream_build", "board_count": len(boards), "rows": 0}

    def on_progress(payload: Dict[str, Any]) -> None:
        crash_ctx.update(
            {
                "stage": "stream_build",
                "board_index": payload.get("board_index"),
                "board_id": payload.get("board_id"),
                "size_class": payload.get("size_class"),
                "ratio": payload.get("ratio"),
                "mask_idx": payload.get("mask_idx"),
                "target": payload.get("target"),
                "rows": payload.get("rows_emitted"),
                "peak_rss_mb": _peak_rss_mb(),
            }
        )

    chunks = iter_masked_ranking_dataset_chunks(boards, config, chunk_rows=max(1, args.shard_rows), progress_hook=on_progress)

    def on_chunk(chunk: pd.DataFrame) -> None:
        nonlocal row_count
        for col in chunk.columns:
            if col not in all_columns:
                all_columns.append(col)
        row_count += int(len(chunk))
        if "group_id" in chunk.columns:
            group_ids.update(str(v) for v in chunk["group_id"].dropna().tolist())
        if "size_class" in chunk.columns:
            for size, sub in chunk.groupby("size_class"):
                size_rows[str(size)] += int(len(sub))
                if "group_id" in sub.columns:
                    size_groups[str(size)].update(str(v) for v in sub["group_id"].dropna().tolist())

        chunk_features = [c for c in chunk.columns if c.startswith("board_state_") or c.startswith("candidate_delta_")]
        for col in chunk_features:
            if col not in stats:
                stats[col] = {"n": 0.0, "sum": 0.0, "sum_sq": 0.0, "counts": {}}
            if col not in primary_cols and is_primary_feature_column(col):
                primary_cols.append(col)
            s = chunk[col].fillna(0.0).astype(float)
            stats[col]["n"] += float(len(s))
            stats[col]["sum"] += float(s.sum())
            stats[col]["sum_sq"] += float((s * s).sum())
            _merge_value_counts(stats[col]["counts"], s.value_counts(dropna=False))
            if "size_class" in chunk.columns:
                for size, sub in chunk.groupby("size_class"):
                    _update_size_agg(size_agg, str(size), col, sub[col])

    try:
        written = write_dataframe_chunks_safe(
            chunks,
            output_path,
            fmt="parquet",
            config=SafeWriteConfig(max_file_mb=args.max_file_mb, producer_script="scripts/build_masked_ranking_dataset.py"),
            on_chunk=on_chunk,
        )
    except Exception:
        if args.debug_crash_report:
            print(json.dumps({"debug_crash_report": crash_ctx}, ensure_ascii=False, indent=2))
        raise

    dead: List[str] = []
    for col in primary_cols:
        item = stats.get(col, {})
        n = float(item.get("n", 0.0))
        std = _std_from_agg(n, float(item.get("sum", 0.0)), float(item.get("sum_sq", 0.0)))
        counts = item.get("counts", {})
        unique = len(counts)
        dom = (max(counts.values()) / max(n, 1.0)) if counts else 1.0
        if unique <= 1 or std <= NEAR_CONSTANT_STD_EPS or dom >= NEAR_CONSTANT_DOMINANT_RATIO:
            dead.append(col)
    dead = sorted(set(dead))
    primary_after = [c for c in primary_cols if c not in set(dead)]

    deprecated = [c for c in [c for c in all_columns if c.startswith("board_state_") or c.startswith("candidate_delta_")] if c not in primary_after]
    schema = {
        "version": FEATURE_SCHEMA_VERSION,
        "schema_strategy": "unified_schema_with_dynamic_optional_bins",
        "feature_columns": primary_after,
        "primary_feature_count_before_pruning": len(primary_cols),
        "primary_feature_count_after_pruning": len(primary_after),
        "near_constant_pruned_features": dead,
        "deprecated_features": deprecated,
        "deprecated_feature_prefixes": list(DEPRECATED_FEATURE_PREFIXES),
        "row_columns": all_columns,
    }
    schema_path = Path(args.feature_schema)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    feature_categories: Dict[str, List[str]] = {
        "target_global_features": [],
        "local_window_features": [],
        "row_col_features": [],
        "neighbor_features": [],
        "interaction_features": [],
        "histogram_features": [],
        "deprecated_features": [],
        "dead_or_near_constant_features": sorted(dead),
        "kept_features": [],
    }
    for col in primary_cols:
        feature_categories.setdefault(_feature_category(col), []).append(col)
    for col in primary_after:
        feature_categories["kept_features"].append(col)

    Path(args.feature_audit_before).parent.mkdir(parents=True, exist_ok=True)
    Path(args.feature_audit_before).write_text(
        json.dumps(
            {"feature_count": len(primary_cols), "feature_categories": feature_categories, "notes": "before pruning/de-dup"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(args.feature_audit_after).write_text(
        json.dumps(
            {
                "feature_count": len(primary_after),
                "removed_features": dead,
                "kept_features": primary_after,
                "notes": "after near-constant pruning",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    merge_report = {
        "removed_features": sorted(dead),
        "merged_features": FEATURE_MERGE_MAP,
        "renamed_features": FEATURE_RENAME_MAP,
        "kept_features": primary_after,
    }
    Path(args.feature_merge_map).write_text(json.dumps(merge_report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.dead_feature_report).write_text(
        json.dumps(
            {
                "threshold_std_eps": NEAR_CONSTANT_STD_EPS,
                "threshold_dominant_ratio": NEAR_CONSTANT_DOMINANT_RATIO,
                "near_constant_features": dead,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    size_summary: Dict[str, Dict[str, int]] = {}
    for size, rows in sorted(size_rows.items()):
        eff = 0
        for col in primary_after:
            agg = size_agg.get(size, {}).get(col)
            if not agg:
                continue
            std = _std_from_agg(float(agg["n"]), float(agg["sum"]), float(agg["sum_sq"]))
            if float(agg["nunique"]) > 0 and std > NEAR_CONSTANT_STD_EPS:
                eff += 1
        size_summary[size] = {
            "rows": int(rows),
            "candidate_groups": int(len(size_groups.get(size, set()))),
            "effective_feature_count": int(eff),
        }

    Path(args.size_aware_summary).write_text(
        json.dumps(
            {
                "schema_strategy": "unified_schema_with_dynamic_optional_bins",
                "size_class_summary": size_summary,
                "primary_feature_count": len(primary_after),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "rows": row_count,
                "groups": len(group_ids),
                "output": args.output,
                "write": written,
                "primary_before": len(primary_cols),
                "primary_after": len(primary_after),
                "dead": len(dead),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
