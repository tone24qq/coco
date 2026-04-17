from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.safe_io import SafeWriteConfig, write_jsonl_records_safe
from src.synthetic_generator import SizeClassProfile, generate_synthetic_from_seed


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/full_boards/full_board_corpus.jsonl")
    parser.add_argument("--profile", default="artifacts/synthetic_generator_profile.json")
    parser.add_argument("--output", default="data/full_boards/synthetic_board_corpus.jsonl")
    parser.add_argument("--per-real", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-file-mb", type=int, default=100)
    args = parser.parse_args()

    real_boards = read_jsonl(Path(args.real_corpus))
    profile_data = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    profiles = {k: SizeClassProfile(**v) for k, v in profile_data.items()}

    rng = random.Random(args.seed)
    out_rows: List[Dict[str, Any]] = []
    for rec in real_boards:
        size_class = rec["size_class"]
        profile = profiles.get(size_class)
        if profile is None:
            raise ValueError(f"missing profile for {size_class}")
        generated = generate_synthetic_from_seed(
            seed_board=rec["grid"],
            profile=profile,
            num_samples=args.per_real,
            rng=rng,
        )
        for idx, (grid, realism) in enumerate(generated):
            out_rows.append(
                {
                    "board_id": f"synth::{rec['board_id']}::{idx:03d}",
                    "lineage_id": str(rec["board_id"]),
                    "rows": rec["rows"],
                    "cols": rec["cols"],
                    "size_class": size_class,
                    "grid": grid,
                    "source": "synthetic_generator_v1",
                    "source_type": "synthetic",
                    "parent_real_size_class": size_class,
                    "parent_board_id": rec["board_id"],
                    "is_real": False,
                    "realism_score": float(realism),
                    "group_id": f"synth::{rec['board_id']}",
                    "issue_id": rec.get("issue_id", rec["board_id"]),
                    "source_file": rec.get("source_file", ""),
                    "order_index": idx,
                }
            )

    out_path = Path(args.output)
    write_jsonl_records_safe(
        out_rows,
        out_path,
        config=SafeWriteConfig(max_file_mb=args.max_file_mb, producer_script="scripts/generate_synthetic_boards.py"),
    )
    print(f"generated {len(out_rows)} synthetic boards -> {out_path}")


if __name__ == "__main__":
    main()
