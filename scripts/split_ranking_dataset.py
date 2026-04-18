from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.safe_io import SafeWriteConfig, read_dataset_auto, write_dataframe_safe


def _bucket(value: str, seed: int) -> float:
    key = f"{seed}:{value}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()[:12]
    return int(h, 16) / float(16**12)


def split_df(
    df: pd.DataFrame,
    holdout_ratio: float,
    split_mode: str,
    seed: int,
    include_synth_in_holdout: bool,
    valid_real_only: bool,
    holdout_real_only: bool,
    exclude_synth_from_valid: bool,
) -> Dict[str, pd.DataFrame]:
    key_col = "board_id" if split_mode == "by_board" else "lineage_id"
    if key_col not in df.columns:
        raise ValueError(f"missing split key column: {key_col}")

    keys = df[[key_col, "source_type"]].drop_duplicates()
    assignments: Dict[str, str] = {}
    for _, row in keys.iterrows():
        source_type = str(row.get("source_type", "real"))
        key = str(row[key_col])
        if source_type == "synthetic" and not include_synth_in_holdout:
            assignments[key] = "train"
            continue
        assignments[key] = "holdout" if _bucket(key, seed) < holdout_ratio else "train"
    if keys.shape[0] > 0 and all(v != "holdout" for v in assignments.values()):
        fallback_key = str(keys.iloc[0][key_col])
        assignments[fallback_key] = "holdout"

    out = df.copy()
    out["split"] = out[key_col].map(assignments)

    holdout = out[out["split"] == "holdout"].copy()
    train_all = out[out["split"] == "train"].copy()

    train_keys = train_all[key_col].drop_duplicates().sort_values().tolist()
    valid_cut = max(1, int(len(train_keys) * 0.1)) if len(train_keys) > 1 else 0
    valid_keys = set(train_keys[:valid_cut])
    valid = train_all[train_all[key_col].isin(valid_keys)].copy()
    train = train_all[~train_all[key_col].isin(valid_keys)].copy()

    if valid_real_only:
        valid = valid[valid["source_type"] == "real"].copy()
    elif exclude_synth_from_valid:
        valid = valid[valid["source_type"] != "synthetic"].copy()

    if holdout_real_only:
        holdout = holdout[holdout["source_type"] == "real"].copy()

    return {"train": train, "valid": valid, "holdout": holdout}


def _stats(frame: pd.DataFrame) -> Dict[str, int]:
    real = frame[frame["source_type"] == "real"] if "source_type" in frame.columns else frame.iloc[0:0]
    synth = frame[frame["source_type"] == "synthetic"] if "source_type" in frame.columns else frame.iloc[0:0]
    return {
        "rows": int(len(frame)),
        "real_rows": int(len(real)),
        "synth_rows": int(len(synth)),
        "real_groups": int(real["group_id"].nunique()) if "group_id" in real.columns and len(real) else 0,
    }


def _per_size_stats(frame: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    if "size_class" not in frame.columns:
        return out
    for size, sub in frame.groupby("size_class"):
        real = sub[sub["source_type"] == "real"] if "source_type" in sub.columns else sub.iloc[0:0]
        out[str(size)] = {
            "rows": int(len(sub)),
            "real_rows": int(len(real)),
            "groups": int(sub["group_id"].nunique()) if "group_id" in sub.columns else 0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["by_board", "by_lineage"], default="by_lineage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-synth-in-holdout", action="store_true")
    parser.add_argument("--valid-real-only", action="store_true")
    parser.add_argument("--holdout-real-only", action="store_true")
    parser.add_argument("--exclude-synth-from-valid", action="store_true")
    parser.add_argument("--max-file-mb", type=int, default=100)
    args = parser.parse_args()

    df = read_dataset_auto(Path(args.dataset_path))
    splits = split_df(
        df,
        args.holdout_ratio,
        args.split_mode,
        args.seed,
        args.include_synth_in_holdout,
        args.valid_real_only,
        args.holdout_real_only,
        args.exclude_synth_from_valid,
    )

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, frame in splits.items():
        out_path = out_root / f"{name}.parquet"
        if frame.empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out_path, index=False)
            meta = {"type": "file", "path": str(out_path), "size_mb": 0.0, "row_count": 0, "columns": list(frame.columns)}
        else:
            meta = write_dataframe_safe(
                frame,
                out_path,
                fmt="parquet",
                config=SafeWriteConfig(max_file_mb=args.max_file_mb, producer_script="scripts/split_ranking_dataset.py"),
                shard_rows=0,
            )
        st = _stats(frame)
        info[name] = {"meta": meta, **st}

    summary = {
        "split_mode": args.split_mode,
        "train_real_rows": info["train"]["real_rows"],
        "train_synth_rows": info["train"]["synth_rows"],
        "valid_real_rows": info["valid"]["real_rows"],
        "valid_synth_rows": info["valid"]["synth_rows"],
        "holdout_real_rows": info["holdout"]["real_rows"],
        "holdout_synth_rows": info["holdout"]["synth_rows"],
        "train_real_groups": info["train"]["real_groups"],
        "valid_real_groups": info["valid"]["real_groups"],
        "holdout_real_groups": info["holdout"]["real_groups"],
        "valid_real_only": args.valid_real_only,
        "holdout_real_only": args.holdout_real_only,
        "exclude_synth_from_valid": args.exclude_synth_from_valid,
        "per_size": {name: _per_size_stats(frame) for name, frame in splits.items()},
    }
    (out_root / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
