from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.snapshots import build_history_snapshot
from src.io.canonical_dataset import load_canonical_or_build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build history snapshot artifacts")
    parser.add_argument(
        "--artifact-mode",
        choices=["runtime", "export"],
        default="runtime",
        help="runtime keeps single parquet; export may shard when size guard triggers",
    )
    args = parser.parse_args()

    canonical = load_canonical_or_build(artifact_mode=args.artifact_mode)
    snapshot, meta = build_history_snapshot(canonical, artifact_mode=args.artifact_mode)
    print(
        json.dumps(
            {
                "status": "ok",
                "artifact_mode": args.artifact_mode,
                "canonical_rows": int(len(canonical)),
                "snapshot_rows": int(len(snapshot)),
                "snapshot_type_counts": meta.get("snapshot_type_counts", {}),
                "paths": meta.get("paths", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
