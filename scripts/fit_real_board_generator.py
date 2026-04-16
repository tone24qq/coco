from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.synthetic_generator import fit_profiles


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-corpus", default="data/full_boards/full_board_corpus.jsonl")
    parser.add_argument("--output", default="artifacts/synthetic_generator_profile.json")
    args = parser.parse_args()

    boards = read_jsonl(Path(args.real_corpus))
    profiles = fit_profiles(boards)
    payload = {k: asdict(v) for k, v in profiles.items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(payload)} size profiles -> {out}")


if __name__ == "__main__":
    main()
