from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from time import perf_counter

import yaml

from src.artifacts import load_artifacts
from src.predict import build_prediction_runtime_state, normalize_predict_config_paths, run_prediction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--config", default="configs/predict.yaml")
    args = p.parse_args()

    cfg = normalize_predict_config_paths(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    artifacts = load_artifacts(Path(cfg["models"]["dir"]))
    state = build_prediction_runtime_state(artifacts, cfg)

    lat = []
    retrieval_lat = []
    model_lat = []
    j20 = []
    cache_hit = 0

    for i in range(args.warmup + args.n):
        t0 = perf_counter()
        out = run_prediction(artifacts, cfg, response_mode="minimal", runtime_state=state)
        elapsed = (perf_counter() - t0) * 1000.0
        meta = out.get("metadata", {})
        if i >= args.warmup:
            lat.append(elapsed)
            lm = meta.get("latency_ms", {})
            retrieval_lat.append(float(lm.get("retrieval", 0.0)))
            model_lat.append(float(lm.get("model_predict", 0.0)))
            if meta.get("recent_cache_status") in {"hit", "stale", "refreshed"}:
                cache_hit += 1
            if meta.get("top20_jaccard_vs_prev") is not None:
                j20.append(float(meta["top20_jaccard_vs_prev"]))

    p95 = statistics.quantiles(lat, n=100)[94] if len(lat) >= 2 else lat[0]
    print(f"warmup={args.warmup}")
    print(f"runs={args.n}")
    print(f"latency_ms_avg={statistics.mean(lat):.3f}")
    print(f"latency_ms_p50={statistics.median(lat):.3f}")
    print(f"latency_ms_p95={p95:.3f}")
    print(f"recent_cache_hit_rate={cache_hit / max(1, args.n):.3f}")
    print(f"retrieval_latency_ms_avg={statistics.mean(retrieval_lat):.3f}")
    print(f"model_latency_ms_avg={statistics.mean(model_lat):.3f}")
    print(f"top20_jaccard_vs_previous_avg={(statistics.mean(j20) if j20 else 0.0):.3f}")


if __name__ == "__main__":
    main()
