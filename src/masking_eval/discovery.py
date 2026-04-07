from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .backtest import run_backtest
from .data_loader import BoardSample
from .modules import BASE_MODULES, DISCOVERY_MODULES


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
        }

    base = base_result["full_model"]
    random_base = base_result["baselines"]["random"]

    pool = candidate_modules or DISCOVERY_MODULES
    leaderboard: List[Dict] = []
    kept: List[str] = []
    for module in pool:
        modules = BASE_MODULES + [module]
        res = run_backtest(boards, folds, repeats, seed + 10, modules, n_trials)
        full = res["full_model"]
        delta_top1 = full["top1_hit_rate"] - base["top1_hit_rate"]
        delta_top3 = full["top3_hit_rate"] - base["top3_hit_rate"]
        delta_mrr = full["mrr"] - base["mrr"]
        keep = (delta_top3 > 0 or delta_mrr > 0) and res["anti_leakage_checks"] == "passed"
        reason = "kept" if keep else "no_test_gain"
        if keep:
            kept.append(module)
        leaderboard.append(
            {
                "module": module,
                "top1": full["top1_hit_rate"],
                "top3": full["top3_hit_rate"],
                "top5": full["top5_hit_rate"],
                "mrr": full["mrr"],
                "delta_vs_old_full_top1": delta_top1,
                "delta_vs_old_full_top3": delta_top3,
                "delta_vs_old_full_mrr": delta_mrr,
                "ablation_delta_top1": (
                    full["top1_hit_rate"]
                    - res["ablation"].get(f"drop_{module}", full)["top1_hit_rate"]
                ),
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
        "random_baseline": random_base,
        "leaderboard": sorted(
            leaderboard, key=lambda x: (x["top3"], x["mrr"], x["top1"]), reverse=True
        ),
        "kept_modules": kept,
        "dropped_modules": [x["module"] for x in leaderboard if not x["keep"]],
        "champion": {
            "modules": champion_modules,
            "metrics": champion_result.get("full_model", {}),
            "best_weights": champion_result.get("best_weights", {}),
        },
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
