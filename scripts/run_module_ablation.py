from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_aggregator_config, load_module_weights
from src.inference_service import _run_inference_detailed


Case = Dict[str, Any]


def _make_case(seed: int, rows: int, cols: int, mask_mod: int = 2) -> Case:
    board = []
    n = 1
    for r in range(rows):
        row = []
        for c in range(cols):
            value = n
            n += 1
            row.append(-1 if (r + c + seed) % mask_mod == 0 else value)
        board.append(row)
    opened = {v for row in board for v in row if v != -1}
    target = min(x for x in range(1, rows * cols + 1) if x not in opened)
    true_cell = ((target - 1) // cols + 1, (target - 1) % cols + 1)
    return {"board": board, "target": target, "true": true_cell, "shape": f"{rows}x{cols}"}


def _metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    n = max(len(rows), 1)
    return {
        "top1_hit_rate": round(sum(1 for r in rows if r["rank"] == 1) / n, 6),
        "top3_hit_rate": round(sum(1 for r in rows if r["rank"] <= 3) / n, 6),
        "top5_hit_rate": round(sum(1 for r in rows if r["rank"] <= 5) / n, 6),
        "mrr": round(sum(1.0 / r["rank"] for r in rows) / n, 6),
        "mean_true_rank": round(sum(r["rank"] for r in rows) / n, 6),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 3),
    }


def _evaluate(cases: List[Case], module_weights: Dict[str, float], fusion_mode: str, agg_type: str = "competitive_ensemble") -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    win_counter: Dict[str, int] = {}
    feature_importance: Dict[str, float] = {}
    disagreements: List[float] = []
    gains: List[float] = []
    for case in cases:
        aggregator_cfg = load_aggregator_config()
        aggregator_cfg["type"] = agg_type
        aggregator_cfg["fusion_mode"] = fusion_mode
        t0 = time.perf_counter()
        out = _run_inference_detailed(
            case["board"],
            case["target"],
            source="ablation",
            module_weights=module_weights,
            apply_reranker_stage=False,
            include_module_details=False,
            aggregator_config=aggregator_cfg,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rank = 999
        for i, cand in enumerate(out["candidate_cells"], start=1):
            if (cand["row"], cand["col"]) == case["true"]:
                rank = i
                break
        rows.append({"rank": rank, "latency_ms": latency_ms})
        top = out["candidate_cells"][0]
        winner = max(top["module_scores"].items(), key=lambda kv: kv[1])[0]
        win_counter[winner] = win_counter.get(winner, 0) + 1
        disagreements.append(float(top.get("disagreement_count", 0.0)))
        gains.append(1.0 / rank)
        for k, v in top.items():
            if k.startswith("module_") and k.endswith("_score"):
                feature_importance[k] = feature_importance.get(k, 0.0) + float(v)
    metrics = _metrics(rows)
    n = max(len(cases), 1)
    return {
        "metrics": metrics,
        "per_module_win_rate": {k: round(v / n, 6) for k, v in sorted(win_counter.items())},
        "judge_selected_feature_importance": {
            k: round(v / n, 6) for k, v in sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:20]
        },
        "disagreement_vs_gain": {
            "avg_disagreement": round(mean(disagreements) if disagreements else 0.0, 6),
            "avg_gain": round(mean(gains) if gains else 0.0, 6),
        },
        "fusion_mode": fusion_mode,
        "aggregator_type": agg_type,
    }


def main() -> None:
    base_weights = load_module_weights()
    modules = list(base_weights.keys())
    cases = [_make_case(s, 8, 10, 2) for s in range(4)] + [_make_case(10 + s, 10, 16, 3) for s in range(4)]

    standalone = {m: _evaluate(cases, {m: 1.0}, "weighted_rank_fusion") for m in modules}
    baseline_equal = _evaluate(cases, {m: 1.0 / len(modules) for m in modules}, "weighted_rank_fusion")
    weighted_rank = _evaluate(cases, base_weights, "weighted_rank_fusion")
    vote_fusion = _evaluate(cases, base_weights, "vote_based_fusion")
    meta_judge = _evaluate(cases, base_weights, "learned_meta_ranker")

    report = {
        "equal_start_competition": baseline_equal,
        "competitive_fusion_weighted_rank": weighted_rank,
        "competitive_fusion_vote": vote_fusion,
        "competitive_fusion_meta_judge": meta_judge,
        "standalone": standalone,
    }
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/module_usefulness_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
