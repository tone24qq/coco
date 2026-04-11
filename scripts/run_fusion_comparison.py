from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_aggregator_config
from src.inference_service import _run_inference_detailed


def _make_case(seed: int, rows: int, cols: int) -> dict:
    n = 1
    board = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = n
            n += 1
            row.append(-1 if (r * 3 + c + seed) % 4 == 0 else v)
        board.append(row)
    opened = {v for row in board for v in row if v != -1}
    target = min(x for x in range(1, rows * cols + 1) if x not in opened)
    true = None
    for r in range(rows):
        for c in range(cols):
            if (r * cols + c + 1) == target:
                true = (r + 1, c + 1)
                break
        if true is not None:
            break
    return {"board": board, "target": target, "true": true}


def _eval(mode: str, cases: List[dict]) -> Dict[str, float]:
    ranks = []
    lats = []
    for case in cases:
        t0 = time.perf_counter()
        out = _run_inference_detailed(
            case["board"],
            case["target"],
            source="fusion",
            module_settings={},
            apply_reranker_stage=False,
            include_module_details=False,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        rank = 999
        for i, cand in enumerate(out["candidate_cells"], start=1):
            if (cand["row"], cand["col"]) == case["true"]:
                rank = i
                break
        ranks.append(rank)
        lats.append(lat)
    n = len(ranks)
    return {
        "top1_hit_rate": round(sum(1 for r in ranks if r == 1) / max(n, 1), 6),
        "top3_hit_rate": round(sum(1 for r in ranks if r <= 3) / max(n, 1), 6),
        "mrr": round(sum(1.0 / r for r in ranks) / max(n, 1), 6),
        "avg_latency_ms": round(sum(lats) / max(n, 1), 3),
    }


def main() -> None:
    cases = [_make_case(i, 8, 10) for i in range(5)] + [_make_case(20 + i, 10, 16) for i in range(4)]
    agg = load_aggregator_config()
    modes = ["weighted_rank_fusion", "vote_based_fusion", "learned_meta_ranker"]
    results = {}
    from src import inference_service as svc

    old_loader = svc.load_aggregator_config

    try:
        for mode in modes:
            def _cfg(mode=mode):
                c = dict(agg)
                c["fusion_mode"] = mode
                return c

            svc.load_aggregator_config = _cfg
            results[mode] = _eval(mode, cases)
    finally:
        svc.load_aggregator_config = old_loader

    best = max(results.items(), key=lambda kv: (kv[1]["top1_hit_rate"], kv[1]["mrr"]))[0]
    report = {
        "modes": results,
        "recommended_default": best,
        "conclusion": {
            "vote_vs_weighted_rank": results["vote_based_fusion"]["mrr"] >= results["weighted_rank_fusion"]["mrr"],
            "meta_vs_weighted_rank": results["learned_meta_ranker"]["mrr"] >= results["weighted_rank_fusion"]["mrr"],
        },
    }
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/fusion_comparison_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
