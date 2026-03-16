from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.snapshots import build_history_snapshot
from src.io.canonical_dataset import load_canonical_or_build


def main() -> None:
    canonical = load_canonical_or_build()
    snapshot, meta = build_history_snapshot(canonical)
    print(
        json.dumps(
            {
                "status": "ok",
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
