from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.io.canonical_dataset import build_canonical_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical dataset from local CSVs"
    )
    parser.add_argument(
        "--artifact-mode",
        choices=["runtime", "export"],
        default="runtime",
        help="runtime keeps single parquet; export may shard when size guard triggers",
    )
    args = parser.parse_args()

    df, audit = build_canonical_dataset(artifact_mode=args.artifact_mode)
    print(
        "saved"
        f" {len(df)} rows -> {audit.get('output_path')}"
        f" | compression={audit.get('selected_compression')}"
        f" | output_format={audit.get('output_format')}"
        f" | missing_years={audit.get('missing_years', [])}"
    )


if __name__ == "__main__":
    main()
