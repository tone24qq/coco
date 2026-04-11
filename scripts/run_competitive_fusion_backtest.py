from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_aggregator_config, load_module_weights
from src.inference_service import _run_inference_detailed


def _load_real_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cases.append(
            {
                "board": row["board"],
                "target": int(row["target_number"]),
                "true": tuple(row["true_cell"]),
                "shape": f"{len(row['board'])}x{len(row['board'][0])}",
            }
        )
    return cases


def _make_synthetic_case(seed: int, rows: int, cols: int, mask_mod: int = 2) -> Dict[str, Any]:
    board = []
    n = 1
    for r in range(rows):
        row = []
        for c in range(cols):
            value = n
            n += 1
            row.append(-1 if (r + c + seed) % mask_mod == 0 else value)
        board.append(row)
    opened = {v for rr in board for v in rr if v != -1}
    target = min(x for x in range(1, rows * cols + 1) if x not in opened)
    true = ((target - 1) // cols + 1, (target - 1) % cols + 1)
    return {"board": board, "target": target, "true": true, "shape": f"{rows}x{cols}"}


def _score_case(case: Dict[str, Any], weights: Dict[str, float], cfg: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    t0 = time.perf_counter()
    out = _run_inference_detailed(
        case["board"],
        case["target"],
        source="competitive_backtest",
        module_weights=weights,
        apply_reranker_stage=False,
        include_module_details=False,
        aggregator_config=cfg,
    )
    latency = (time.perf_counter() - t0) * 1000.0
    rank = 999
    for i, cand in enumerate(out["candidate_cells"], start=1):
        if (cand["row"], cand["col"]) == case["true"]:
            rank = i
            break
    return rank, latency, out


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


def _run_modes(cases: List[Dict[str, Any]], base_cfg: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    mode_rows: Dict[str, List[Dict[str, float]]] = {
        "current_hierarchical_fusion": [],
        "equal_weight_average": [],
        "weighted_rank_fusion": [],
        "vote_based_fusion": [],
        "learned_meta_ranker": [],
    }
    fold_results: Dict[str, List[int]] = {k: [] for k in mode_rows}
    disagreement_vs_gain: Dict[str, Dict[str, float]] = {}
    for mode_name, fusion_mode, agg_type, use_equal_weights in [
        ("current_hierarchical_fusion", "weighted_rank_fusion", "gate_then_weighted_sum", False),
        ("equal_weight_average", "weighted_rank_fusion", "competitive_ensemble", True),
        ("weighted_rank_fusion", "weighted_rank_fusion", "competitive_ensemble", False),
        ("vote_based_fusion", "vote_based_fusion", "competitive_ensemble", False),
        ("learned_meta_ranker", "learned_meta_ranker", "competitive_ensemble", False),
    ]:
        gains: List[float] = []
        disagrees: List[float] = []
        for case in cases:
            cfg = dict(base_cfg)
            cfg["type"] = agg_type
            cfg["fusion_mode"] = fusion_mode
            w = {k: 1.0 / len(weights) for k in weights} if use_equal_weights else weights
            rank, latency, out = _score_case(case, w, cfg)
            mode_rows[mode_name].append({"rank": rank, "latency_ms": latency})
            fold_results[mode_name].append(rank)
            gains.append(1.0 / rank)
            if out.get("candidate_cells"):
                top = out["candidate_cells"][0]
                disagrees.append(float(top.get("disagreement_count", 0.0)))
            else:
                disagrees.append(0.0)
        disagreement_vs_gain[mode_name] = {
            "avg_disagreement": round(sum(disagrees) / max(len(disagrees), 1), 6),
            "avg_gain": round(sum(gains) / max(len(gains), 1), 6),
        }
    mode_win_counts = {k: sum(1 for x in v if x["rank"] == 1) for k, v in mode_rows.items()}
    return {
        "comparison": {k: _metrics(v) for k, v in mode_rows.items()},
        "fold_wise_results": fold_results,
        "disagreement_vs_gain": disagreement_vs_gain,
        "mode_win_counts": mode_win_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", default="data/competitive_cases.jsonl")
    parser.add_argument("--min-real-cases", type=int, default=8)
    args = parser.parse_args()

    real_cases = _load_real_cases(Path(args.real_data))
    synthetic_cases = [_make_synthetic_case(i, 8, 10, 2) for i in range(4)] + [
        _make_synthetic_case(100 + i, 10, 16, 3) for i in range(4)
    ]

    base_cfg = load_aggregator_config()
    weights = load_module_weights()

    real_results = _run_modes(real_cases, base_cfg, weights) if real_cases else {"comparison": {}}
    synthetic_results = _run_modes(synthetic_cases, base_cfg, weights)

    insufficient_real_data = len(real_cases) < args.min_real_cases
    recommended_default = "weighted_rank_fusion"
    rationale = "default_deterministic_baseline"
    if not insufficient_real_data and real_results["comparison"]:
        preference = {
            "weighted_rank_fusion": 0,
            "learned_meta_ranker": 1,
            "vote_based_fusion": 2,
            "equal_weight_average": 3,
            "current_hierarchical_fusion": 4,
        }
        ordered = sorted(
            real_results["comparison"].items(),
            key=lambda kv: (kv[1]["mean_true_rank"], -kv[1]["top1_hit_rate"], preference.get(kv[0], 99)),
        )
        recommended_default = ordered[0][0]
        rationale = "chosen_by_real_data_comparison"

    artifact_path = Path("artifacts/competitive_judge_artifact.json")
    if artifact_path.exists():
        judge_feature_importance = json.loads(artifact_path.read_text(encoding="utf-8")).get("coef", [])
    else:
        judge_feature_importance = []
    report = {
        "real_case_count": len(real_cases),
        "insufficient_real_data": insufficient_real_data,
        "real_data_comparison": real_results,
        "synthetic_smoke_comparison": synthetic_results,
        "judge_feature_importance": judge_feature_importance,
        "recommended_default": recommended_default,
        "recommended_default_rationale": rationale,
    }

    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/competitive_fusion_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
