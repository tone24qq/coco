from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.masking_dataset import MaskingConfig, build_masked_ranking_dataset, write_rank_dataset
from src.whole_board_features import DEPRECATED_FEATURE_PREFIXES, is_primary_feature_column


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/full_boards/full_board_corpus.jsonl")
    parser.add_argument("--synthetic-corpus", default="data/full_boards/synthetic_board_corpus.jsonl")
    parser.add_argument("--output", default="data/ranking/ranking_dataset.parquet")
    parser.add_argument("--mask-ratios", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--masks-per-ratio", type=int, default=2)
    parser.add_argument("--shard-rows", type=int, default=0)
    parser.add_argument("--feature-schema", default="artifacts/feature_schema.json")
    parser.add_argument("--max-file-mb", type=int, default=100)
    args = parser.parse_args()

    real_rows = read_jsonl(Path(args.real_corpus))
    synth_rows: List[Dict[str, Any]] = []
    synth_path = Path(args.synthetic_corpus)
    if synth_path.exists():
        synth_rows = read_jsonl(synth_path)

    boards = real_rows + synth_rows
    ratios = [float(x.strip()) for x in args.mask_ratios.split(",") if x.strip()]
    df = build_masked_ranking_dataset(boards, MaskingConfig(ratios=ratios, masks_per_ratio=args.masks_per_ratio))
    written = write_rank_dataset(
        df,
        Path(args.output),
        shard_rows=args.shard_rows,
        max_file_mb=args.max_file_mb,
        producer_script="scripts/build_masked_ranking_dataset.py",
    )

    feature_cols = [c for c in df.columns if is_primary_feature_column(c)]
    deprecated = [
        c
        for c in df.columns
        if c.startswith("board_state_") or c.startswith("candidate_delta_")
        if not is_primary_feature_column(c)
    ]
    schema = {
        "version": "whole_board_features_v2_residue_multiple10",
        "feature_columns": feature_cols,
        "deprecated_features": deprecated,
        "deprecated_feature_prefixes": list(DEPRECATED_FEATURE_PREFIXES),
        "row_columns": list(df.columns),
    }
    schema_path = Path(args.feature_schema)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"rows": len(df), "output": args.output, "write": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
