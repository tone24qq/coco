from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.io.canonical_dataset import build_canonical_dataset  # noqa: E402


def main() -> None:
    df, audit = build_canonical_dataset()
    print(
        "saved"
        f" {len(df)} rows -> data/processed/bingo_draws_canonical.csv"
        f" | missing_years={audit.get('missing_years', [])}"
    )


if __name__ == "__main__":
    main()
