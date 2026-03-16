from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.predict import Predictor


def _one_run(predictor: Predictor, df: pd.DataFrame) -> int:
    t0 = time.perf_counter()
    predictor.predict_from_draws(df, min_history=min(201, len(df) - 1))
    return int((time.perf_counter() - t0) * 1000)


def main() -> None:
    predictor = Predictor.load()
    sample = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "bingo_draws.csv").tail(
        500
    )
    sample = sample.reset_index(drop=True)

    latencies = [_one_run(predictor, sample) for _ in range(20)]
    out = {
        "runs": len(latencies),
        "mean_ms": float(statistics.mean(latencies)),
        "p50_ms": float(statistics.median(latencies)),
        "max_ms": float(max(latencies)),
        "min_ms": float(min(latencies)),
    }
    out_path = PROJECT_ROOT / "reports" / "predict_latency_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"latency": out, "report": str(out_path)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
