from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.io.canonical_dataset import load_canonical_or_build
from src.predict import Predictor


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {
        "exact_hit@3": float(df["exact3"].mean()),
        "exact_hit@10": float(df["exact10"].mean()),
        "top3_at_least_one_exact": float(df["top3_any"].mean()),
        "adj_hit_pm1@3": float(df["adj3"].mean()),
        "adj_hit_pm1@10": float(df["adj10"].mean()),
        "mean_min_distance_at_3": float(df["min_dist3"].mean()),
        "latency_ms_mean": float(df["latency_ms"].mean()),
    }


def _evaluate_variant(predictor: Predictor, draws: list[list[int]], name: str) -> dict:
    start_idx = 260
    rows = []
    max_samples = 12
    for i in range(start_idx, min(len(draws) - 1, start_idx + max_samples)):
        hist = draws[: i + 1]
        next_draw = set(draws[i + 1])
        df = pd.DataFrame(
            {
                "issue": list(range(1, len(hist) + 1)),
                "draw_date": ["2026-01-01" for _ in hist],
                "numbers": [json.dumps(d, ensure_ascii=False) for d in hist],
            }
        )
        t0 = time.perf_counter()
        out = predictor.predict_from_draws(df, min_history=min(201, len(hist) - 1))
        latency_ms = int((time.perf_counter() - t0) * 1000)

        top3 = [int(x) for x in out["top3_numbers"]]
        top10 = [int(x) for x in out["top10_numbers"]]

        rows.append(
            {
                "exact3": float(sum(1 for n in top3 if n in next_draw) / 3.0),
                "exact10": float(sum(1 for n in top10 if n in next_draw) / 10.0),
                "top3_any": float(any(n in next_draw for n in top3)),
                "adj3": float(
                    sum(1 for n in top3 if any(abs(n - t) <= 1 for t in next_draw))
                    / 3.0
                ),
                "adj10": float(
                    sum(1 for n in top10 if any(abs(n - t) <= 1 for t in next_draw))
                    / 10.0
                ),
                "min_dist3": float(
                    np.mean([min(abs(n - t) for t in next_draw) for n in top3])
                ),
                "latency_ms": latency_ms,
            }
        )

    return {"variant": name, "metrics": _metrics(rows), "samples": len(rows)}


def _variant_predictor(
    base: Predictor, *, long_feat: bool, history: bool, rerank: bool
) -> Predictor:
    p = deepcopy(base)
    p.runtime_config = deepcopy(base.runtime_config)
    p.runtime_config.setdefault("long_feature_injection", {})
    p.runtime_config["long_feature_injection"]["enabled"] = bool(long_feat)
    p.runtime_config.setdefault("history_prior", {})
    p.runtime_config["history_prior"]["enabled"] = bool(history)
    p.runtime_config.setdefault("analysis_rerank", {})
    p.runtime_config["analysis_rerank"]["enabled"] = bool(rerank)
    return p


def main() -> None:
    canonical = (
        load_canonical_or_build().sort_values("issue").tail(1200).reset_index(drop=True)
    )
    draws = [sorted(json.loads(x)) for x in canonical["numbers"].astype(str).tolist()]

    base = Predictor.load()
    variants = [
        (
            "A_baseline",
            _variant_predictor(base, long_feat=False, history=False, rerank=False),
        ),
        (
            "B_long_features",
            _variant_predictor(base, long_feat=True, history=False, rerank=False),
        ),
        (
            "C_long_features_plus_history_blend",
            _variant_predictor(base, long_feat=True, history=True, rerank=False),
        ),
        (
            "D_plus_analysis_rerank",
            _variant_predictor(base, long_feat=True, history=True, rerank=True),
        ),
    ]

    results = [_evaluate_variant(pred, draws, name) for name, pred in variants]
    out_path = PROJECT_ROOT / "reports" / "predict_ablation_report.json"
    out_path.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"results": results, "report": str(out_path)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
