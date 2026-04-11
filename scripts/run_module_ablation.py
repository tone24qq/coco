from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_module_weights
from src.inference_service import _run_inference_detailed

Cell = Tuple[int, int]


def _make_case(seed: int, rows: int, cols: int, mask_mod: int = 2) -> dict:
    board = []
    n = 1
    for r in range(rows):
        row = []
        for c in range(cols):
            v = n
            n += 1
            if (r + c + seed) % mask_mod == 0:
                row.append(-1)
            else:
                row.append(v)
        board.append(row)
    opened = {v for row in board for v in row if v != -1}
    target = min(x for x in range(1, rows * cols + 1) if x not in opened)
    true = None
    for r in range(rows):
        for c in range(cols):
            if (r + c + seed) % mask_mod == 0 and (r * cols + c + 1) == target:
                true = (r + 1, c + 1)
                break
        if true is not None:
            break
    return {"board": board, "target": target, "true": true}


def _rank_metrics(rows: List[dict]) -> Dict[str, float]:
    n = len(rows)
    top1 = sum(1 for x in rows if x["rank"] == 1) / max(n, 1)
    top3 = sum(1 for x in rows if x["rank"] <= 3) / max(n, 1)
    top5 = sum(1 for x in rows if x["rank"] <= 5) / max(n, 1)
    mrr = sum(1.0 / x["rank"] for x in rows) / max(n, 1)
    mean_rank = sum(x["rank"] for x in rows) / max(n, 1)
    mean_lat = sum(x["latency_ms"] for x in rows) / max(n, 1)
    return {
        "top1_hit_rate": round(top1, 6),
        "top3_hit_rate": round(top3, 6),
        "top5_hit_rate": round(top5, 6),
        "mrr": round(mrr, 6),
        "mean_true_rank": round(mean_rank, 6),
        "avg_latency_ms": round(mean_lat, 3),
    }


def _eval_cases(cases: List[dict], weights: Dict[str, float]) -> Dict[str, float]:
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        out = _run_inference_detailed(
            case["board"],
            case["target"],
            source="ablation",
            module_weights=weights,
            apply_reranker_stage=False,
            include_module_details=False,
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        rank = 999
        if case["true"] is not None:
            for i, cand in enumerate(out["candidate_cells"], start=1):
                if (cand["row"], cand["col"]) == case["true"]:
                    rank = i
                    break
        rows.append({"rank": rank, "latency_ms": lat_ms})
    return _rank_metrics(rows)


def _weight_of(module: str, base: Dict[str, float]) -> Dict[str, float]:
    return {module: 1.0} if module in base else {module: 1.0}


def main() -> None:
    base_weights = load_module_weights()
    modules = list(base_weights.keys())
    new_modules = {
        "focus_score",
        "connectivity_heatmap",
        "difference_trend",
        "skip_patterns",
        "mirror_sequences",
        "tail_analyzer",
    }
    cases = []
    for s in range(4):
        cases.append(_make_case(s, 8, 10, mask_mod=2))
    for s in range(3):
        cases.append(_make_case(10 + s, 10, 16, mask_mod=3))

    standalone = {m: _eval_cases(cases, _weight_of(m, base_weights)) for m in modules}
    baseline = _eval_cases(cases, base_weights)

    plus = {}
    minus = {}
    core_modules = [m for m in modules if m not in new_modules]
    for m in new_modules:
        if m not in modules:
            continue
        w = dict(base_weights)
        w[m] = max(w.get(m, 0.0), 0.05)
        plus[m] = _eval_cases(cases, w)
    for m in core_modules:
        w = dict(base_weights)
        w.pop(m, None)
        if not w:
            continue
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        minus[m] = _eval_cases(cases, w)

    useful, neutral, dragging = [], [], []
    for m, stats in standalone.items():
        gain = stats["top1_hit_rate"] - baseline["top1_hit_rate"]
        if gain > 0.02:
            useful.append(m)
        elif gain < -0.02:
            dragging.append(m)
        else:
            neutral.append(m)

    report = {
        "baseline": baseline,
        "standalone": standalone,
        "incremental_plus": plus,
        "incremental_minus": minus,
        "pareto": {
            "useful_modules": useful,
            "neutral_modules": neutral,
            "dragging_modules": dragging,
            "expensive_but_helpful": [m for m, s in standalone.items() if s["avg_latency_ms"] > 120 and m in useful],
            "expensive_and_not_helpful": [m for m, s in standalone.items() if s["avg_latency_ms"] > 120 and m in dragging],
        },
    }
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/module_usefulness_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
