#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_module_weights
from src.mainline_eval import (
    CORE_MODULES,
    discover_full_boards_with_audit,
    random_weight_candidates,
    run_weighted_eval,
    write_csv,
)
from src.inference_config import load_module_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="gogo")
    parser.add_argument("--output-dir", default="reports/mainline_eval")
    parser.add_argument("--masking-ratio", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight-trials", type=int, default=20)
    parser.add_argument("--apply-reranker-stage", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    discovery = discover_full_boards_with_audit(input_dir)
    boards = discovery.boards
    baseline_weights = load_module_weights()
    module_settings_snapshot = load_module_settings()

    search_rows = []
    best = None
    candidates = [baseline_weights] + random_weight_candidates(CORE_MODULES, args.weight_trials, args.seed)
    for idx, weights in enumerate(candidates):
        per_case, metrics = run_weighted_eval(
            boards=boards,
            weights=weights,
            masking_ratio=args.masking_ratio,
            repeats=args.repeats,
            seed=args.seed + idx * 1000,
            apply_reranker_stage=args.apply_reranker_stage,
        )
        search_rows.append({"trial": idx, "weights": json.dumps(weights, ensure_ascii=False), **metrics})
        score_key = (metrics["top1_hit_rate"], metrics["top3_hit_rate"], metrics["mrr"], -metrics["mean_rank"])
        if best is None or score_key > best[0]:
            best = (score_key, weights, metrics, per_case)

    assert best is not None
    best_weights = best[1]
    best_metrics = best[2]
    best_per_case = best[3]
    _, baseline_metrics = run_weighted_eval(
        boards=boards,
        weights=baseline_weights,
        masking_ratio=args.masking_ratio,
        repeats=args.repeats,
        seed=args.seed,
        apply_reranker_stage=args.apply_reranker_stage,
    )

    ablation_rows = []
    for module in CORE_MODULES:
        ablated = {k: v for k, v in best_weights.items() if k != module}
        total = sum(ablated.values())
        if total > 0:
            ablated = {k: v / total for k, v in ablated.items()}
        _, metrics = run_weighted_eval(
            boards=boards,
            weights=ablated,
            masking_ratio=args.masking_ratio,
            repeats=args.repeats,
            seed=args.seed + 5000,
            apply_reranker_stage=args.apply_reranker_stage,
        )
        ablation_rows.append({"drop_module": module, "weights": json.dumps(ablated, ensure_ascii=False), **metrics})

    write_csv(output_dir / "per_case_results.csv", best_per_case)
    write_csv(output_dir / "weight_search_results.csv", search_rows)
    write_csv(output_dir / "ablation_results.csv", ablation_rows)
    summary = {
        "best_weights": best_weights,
        "best_metric_summary": best_metrics,
        "baseline_metric_summary": baseline_metrics,
        "delta_vs_baseline": {k: best_metrics[k] - baseline_metrics.get(k, 0.0) for k in best_metrics},
        "num_boards": len(boards),
        "invalid_board_count": int(sum(discovery.invalid_reasons.values())),
        "invalid_reasons": discovery.invalid_reasons,
        "apply_reranker_stage": bool(args.apply_reranker_stage),
        "module_settings_snapshot": module_settings_snapshot,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
