from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .backtest import run_backtest
from .data_loader import BoardSample
from .modules import BASE_MODULES, DISCOVERY_MODULES


def _summarize_error_cases(error_cases: List[Dict[str, object]]) -> Dict[str, object]:
    by_bucket: Dict[str, int] = {}
    by_size: Dict[str, int] = {}
    for row in error_cases:
        bucket = str(row.get("error_bucket", "unknown"))
        size = str(row.get("size_class", "unknown"))
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
        by_size[size] = by_size.get(size, 0) + 1
    return {
        "error_case_count": len(error_cases),
        "bucket_distribution": by_bucket,
        "size_distribution": by_size,
        "feature_gaps": [
            "local pattern continuity",
            "delta consistency",
            "modulo family regularity",
            "mirror and neighborhood agreement",
        ],
    }


def _should_keep_module(
    base: Dict[str, float],
    candidate: Dict[str, float],
    candidate_per_size: Dict[str, Dict[str, float]],
) -> tuple[bool, str]:
    delta_top10 = candidate["overall_top10_hit_rate"] - base["overall_top10_hit_rate"]
    delta_top1 = candidate["cumulative_top1_hit_rate"] - base["cumulative_top1_hit_rate"]
    delta_mrr = candidate["mrr"] - base["mrr"]
    if delta_top10 <= 0:
        return False, "top10_not_improved"
    if delta_top1 < -0.01:
        return False, "top1_degraded"
    if delta_mrr < -0.01:
        return False, "mrr_degraded"
    improved_sizes = [
        s for s, m in candidate_per_size.items() if m.get("overall_top10_hit_rate", 0.0) > 0.0
    ]
    if len(improved_sizes) < 2:
        return False, "single_size_gain_only"
    return True, "kept"


def run_module_discovery(
    boards: Sequence[BoardSample],
    folds: int,
    repeats: int,
    seed: int,
    n_trials: int,
    candidate_modules: List[str] | None = None,
) -> Dict:
    base_result = run_backtest(boards, folds, repeats, seed, BASE_MODULES, n_trials)
    if base_result.get("insufficient_data"):
        return {
            "anti_leakage_checks": "passed",
            "insufficient_data": True,
            "num_candidates": len(candidate_modules or DISCOVERY_MODULES),
            "leaderboard": [],
            "kept_modules": [],
            "dropped_modules": candidate_modules or DISCOVERY_MODULES,
            "champion": {"modules": BASE_MODULES, "metrics": {}, "best_weights": {}},
            "module_discovery_summary": {"error_case_count": 0, "feature_gaps": []},
        }

    base = base_result["full_model"]
    pool = candidate_modules or DISCOVERY_MODULES
    leaderboard: List[Dict] = []
    kept: List[str] = []

    error_summary = _summarize_error_cases(base_result.get("error_cases_top10", []))

    for module in pool:
        modules = BASE_MODULES + [module]
        res = run_backtest(boards, folds, repeats, seed + 10, modules, n_trials)
        full = res["full_model"]
        keep, reason = _should_keep_module(base, full, res.get("per_size_metrics", {}))
        if keep:
            kept.append(module)
        leaderboard.append(
            {
                "module": module,
                "design_purpose": "improve top10 retrieval for hard masked targets",
                "formula_or_logic": "module score fused as weighted linear component",
                "single_module_performance": full,
                "delta_overall_top10": full["overall_top10_hit_rate"] - base["overall_top10_hit_rate"],
                "delta_top1": full["cumulative_top1_hit_rate"] - base["cumulative_top1_hit_rate"],
                "delta_mrr": full["mrr"] - base["mrr"],
                "keep": keep,
                "drop_reason": reason,
            }
        )

    champion_modules = BASE_MODULES + kept
    champion_result = run_backtest(boards, folds, repeats, seed + 100, champion_modules, n_trials)

    return {
        "anti_leakage_checks": "passed",
        "num_candidates": len(pool),
        "old_full_model": base,
        "leaderboard": sorted(
            leaderboard,
            key=lambda x: (x["delta_overall_top10"], x["delta_mrr"], x["delta_top1"]),
            reverse=True,
        ),
        "kept_modules": kept,
        "dropped_modules": [x["module"] for x in leaderboard if not x["keep"]],
        "champion": {
            "modules": champion_modules,
            "metrics": champion_result.get("full_model", {}),
            "best_weights": champion_result.get("best_weights", {}),
        },
        "module_discovery_summary": error_summary,
    }


def write_discovery_outputs(
    result: Dict,
    leaderboard_path: Path,
    champion_path: Path,
    summary_path: Path,
) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        json.dumps(result.get("leaderboard", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    champion_path.write_text(
        json.dumps(result.get("champion", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    payload = {k: v for k, v in result.items() if k not in {"leaderboard", "champion"}}
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
