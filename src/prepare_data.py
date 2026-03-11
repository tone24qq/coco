from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import shutil
from pathlib import Path

import pandas as pd  # noqa: E402

from src.utils import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    RAW_FILES,
    ROOT,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    frames = []
    for file_name in RAW_FILES:
        src = ROOT / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing raw file: {src}")
        dst = DATA_RAW_DIR / file_name
        shutil.copy2(src, dst)
        df = pd.read_csv(dst)
        issue_col = "期別" if "期別" in df.columns else "issue"
        date_col = "開獎日期" if "開獎日期" in df.columns else "draw_date"
        number_cols = [f"獎號{i}" for i in range(1, 21)]
        if not all(col in df.columns for col in number_cols):
            number_cols = [f"n{i}" for i in range(1, 21)]
        cleaned = pd.DataFrame(
            {
                "issue": pd.to_numeric(df[issue_col], errors="coerce").astype("Int64"),
                "draw_date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime(
                    "%Y-%m-%d"
                ),
            }
        )
        numbers = df[number_cols].apply(
            lambda r: sorted([int(v) for v in r.tolist()]), axis=1
        )
        cleaned["numbers"] = numbers.apply(lambda v: json.dumps(v, ensure_ascii=False))
        cleaned = cleaned.dropna(subset=["issue", "draw_date"]).copy()
        frames.append(cleaned)

    result = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["issue"])
        .sort_values("issue")
    )
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(DATA_PROCESSED_DIR / "bingo_draws.csv", index=False)
    print(f"saved {len(result)} rows -> {DATA_PROCESSED_DIR / 'bingo_draws.csv'}")


if __name__ == "__main__":
    main()
