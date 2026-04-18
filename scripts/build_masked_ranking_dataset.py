from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.masking_dataset import MaskingConfig, build_masked_ranking_dataset, write_rank_dataset
from src.whole_board_features import (
    DEPRECATED_FEATURE_PREFIXES,
    FEATURE_MERGE_MAP,
    FEATURE_RENAME_MAP,
    FEATURE_SCHEMA_VERSION,
    NEAR_CONSTANT_DOMINANT_RATIO,
    NEAR_CONSTANT_STD_EPS,
    is_primary_feature_column,
)


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


def _dead_or_constant(df: pd.DataFrame, features: List[str]) -> List[str]:
    dead: List[str] = []
    for col in features:
        s = df[col].fillna(0.0)
        if s.empty:
            dead.append(col)
            continue
        unique = s.nunique(dropna=False)
        if unique <= 1:
            dead.append(col)
            continue
        if float(s.std()) <= NEAR_CONSTANT_STD_EPS:
            dead.append(col)
            continue
        dominant = float(s.value_counts(normalize=True, dropna=False).iloc[0])
        if dominant >= NEAR_CONSTANT_DOMINANT_RATIO:
            dead.append(col)
    return sorted(set(dead))


def _size_summary(df: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for size, sub in df.groupby("size_class"):
        valid = []
        for col in features:
            if col not in sub.columns:
                continue
            s = sub[col].fillna(0.0)
            if s.nunique(dropna=False) > 1 and float(s.std()) > NEAR_CONSTANT_STD_EPS:
                valid.append(col)
        out[str(size)] = {
            "rows": int(len(sub)),
            "candidate_groups": int(sub["group_id"].nunique()) if "group_id" in sub.columns else 0,
            "effective_feature_count": int(len(valid)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/full_boards/full_board_corpus.jsonl")
    parser.add_argument("--synthetic-corpus", default="data/full_boards/synthetic_board_corpus.jsonl")
    parser.add_argument("--output", default="data/ranking/ranking_dataset.parquet")
    parser.add_argument("--mask-ratios", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--masks-per-ratio", type=int, default=2)
    parser.add_argument("--shard-rows", type=int, default=0)
    parser.add_argument("--feature-schema", default="artifacts/feature_schema_residue.json")
    parser.add_argument("--max-file-mb", type=int, default=100)
    parser.add_argument("--feature-audit-before", default="reports/feature_audit_before.json")
    parser.add_argument("--feature-audit-after", default="reports/feature_audit_after.json")
    parser.add_argument("--feature-merge-map", default="reports/feature_merge_map.json")
    parser.add_argument("--dead-feature-report", default="reports/dead_feature_report.json")
    parser.add_argument("--size-aware-summary", default="reports/size_aware_feature_summary.json")
    args = parser.parse_args()

    real_rows = read_jsonl(Path(args.real_corpus))
    synth_rows: List[Dict[str, Any]] = []
    synth_path = Path(args.synthetic_corpus)
    if synth_path.exists():
        synth_rows = read_jsonl(synth_path)

    boards = real_rows + synth_rows
    ratios = [float(x.strip()) for x in args.mask_ratios.split(",") if x.strip()]
    df = build_masked_ranking_dataset(boards, MaskingConfig(ratios=ratios, masks_per_ratio=args.masks_per_ratio))

    all_feature_cols = [
        c for c in df.columns if c.startswith("board_state_") or c.startswith("candidate_delta_")
    ]
    primary_before = [c for c in all_feature_cols if is_primary_feature_column(c)]
    dead = _dead_or_constant(df, primary_before)
    primary_after = [c for c in primary_before if c not in set(dead)]

    written = write_rank_dataset(
        df,
        Path(args.output),
        shard_rows=args.shard_rows,
        max_file_mb=args.max_file_mb,
        producer_script="scripts/build_masked_ranking_dataset.py",
    )

    deprecated = [c for c in all_feature_cols if c not in primary_after]
    schema = {
        "version": FEATURE_SCHEMA_VERSION,
        "schema_strategy": "unified_schema_with_dynamic_optional_bins",
        "feature_columns": primary_after,
        "primary_feature_count_before_pruning": len(primary_before),
        "primary_feature_count_after_pruning": len(primary_after),
        "near_constant_pruned_features": dead,
        "deprecated_features": deprecated,
        "deprecated_feature_prefixes": list(DEPRECATED_FEATURE_PREFIXES),
        "row_columns": list(df.columns),
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
    for col in primary_before:
        cat = _feature_category(col)
        feature_categories.setdefault(cat, []).append(col)
        if col in dead and col not in feature_categories["dead_or_near_constant_features"]:
            feature_categories["dead_or_near_constant_features"].append(col)

    for col in primary_after:
        feature_categories["kept_features"].append(col)

    Path(args.feature_audit_before).parent.mkdir(parents=True, exist_ok=True)
    Path(args.feature_audit_before).write_text(
        json.dumps(
            {
                "feature_count": len(primary_before),
                "feature_categories": feature_categories,
                "notes": "before pruning/de-dup",
            },
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

    removed_map = {name: "near_constant_or_dead" for name in dead}
    merge_report = {
        "removed_features": sorted(dead),
        "merged_features": FEATURE_MERGE_MAP,
        "renamed_features": FEATURE_RENAME_MAP,
        "kept_features": primary_after,
        "why_removed_or_merged": {
            **removed_map,
            **{k: v for k, v in FEATURE_MERGE_MAP.items()},
            **{k: f"renamed_to:{v}" for k, v in FEATURE_RENAME_MAP.items()},
        },
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

    size_summary = _size_summary(df, primary_after)
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
                "rows": len(df),
                "output": args.output,
                "write": written,
                "primary_before": len(primary_before),
                "primary_after": len(primary_after),
                "dead": len(dead),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
