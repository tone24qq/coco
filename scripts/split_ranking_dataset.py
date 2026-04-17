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


def split_df(df: pd.DataFrame, holdout_ratio: float, split_mode: str, seed: int, include_synth_in_holdout: bool) -> Dict[str, pd.DataFrame]:
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

    out = df.copy()
    out["split"] = out[key_col].map(assignments)
    holdout = out[out["split"] == "holdout"].copy()
    train_all = out[out["split"] == "train"].copy()

    train_keys = train_all[key_col].drop_duplicates().sort_values().tolist()
    valid_cut = max(1, int(len(train_keys) * 0.1)) if len(train_keys) > 1 else 0
    valid_keys = set(train_keys[:valid_cut])
    valid = train_all[train_all[key_col].isin(valid_keys)].copy()
    train = train_all[~train_all[key_col].isin(valid_keys)].copy()
    return {"train": train, "valid": valid, "holdout": holdout}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["by_board", "by_lineage"], default="by_board")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-synth-in-holdout", action="store_true")
    parser.add_argument("--max-file-mb", type=int, default=100)
    args = parser.parse_args()

    df = read_dataset_auto(Path(args.dataset_path))
    splits = split_df(df, args.holdout_ratio, args.split_mode, args.seed, args.include_synth_in_holdout)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, frame in splits.items():
        meta = write_dataframe_safe(
            frame,
            out_root / f"{name}.parquet",
            fmt="parquet",
            config=SafeWriteConfig(max_file_mb=args.max_file_mb, producer_script="scripts/split_ranking_dataset.py"),
            shard_rows=0,
        )
        info[name] = {"rows": int(len(frame)), "meta": meta}

    (out_root / "split_summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["rows"] for k, v in info.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
