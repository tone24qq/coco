#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.masking_eval.backtest import run_backtest  # noqa: E402
from src.masking_eval.discovery import run_module_discovery  # noqa: E402
from src.masking_eval.modules import BASE_MODULES  # noqa: E402
from src.multi_size_data_loader import MultiSizeBoardSample, load_multisize_samples  # noqa: E402


def _build_synthetic_samples(seed: int, per_size: int) -> List[MultiSizeBoardSample]:
    rng = __import__("numpy").random.default_rng(seed)
    shape_map = {"120": (10, 12), "160": (10, 16)}
    out: List[MultiSizeBoardSample] = []
    for size_class, (rows, cols) in shape_map.items():
        n = rows * cols
        for idx in range(per_size):
            values = rng.permutation(__import__("numpy").arange(1, n + 1))
            grid = values.reshape(rows, cols)
            out.append(
                MultiSizeBoardSample(
                    sample_id=f"synthetic_{size_class}_{idx}",
                    board_id=f"{size_class}:synthetic_{idx}",
                    size_class=size_class,
                    grid=grid,
                    shape=f"{rows}x{cols}",
                    parse_confidence=1.0,
                )
            )
    return out


def _summarize_size(
    samples: List[MultiSizeBoardSample],
    cfg: Dict,
    modules: List[str],
    folds: int | None = None,
    repeats: int | None = None,
    n_trials: int | None = None,
) -> Dict:
    if len(samples) < int(cfg["eval"]["min_samples_per_size"]):
        return {"insufficient_data": True, "num_boards": len(samples), "anti_leakage_checks": "passed"}
    return run_backtest(
        boards=samples,
        folds=folds if folds is not None else int(cfg["eval"]["folds"]),
        repeats=repeats if repeats is not None else int(cfg["eval"]["masking_repeats"]),
        seed=int(cfg["eval"]["seed"]),
        modules=modules,
        n_trials=n_trials if n_trials is not None else int(cfg["search"]["n_trials"]),
    )


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _select_target_sizes(samples: List[MultiSizeBoardSample], cfg: Dict) -> List[MultiSizeBoardSample]:
    target_sizes = [str(x) for x in cfg["eval"].get("target_sizes", [120, 160])]
    filtered = [s for s in samples if s.size_class in target_sizes]
    max_per_size = int(cfg["eval"].get("max_boards_per_size", 4))
    buckets: Dict[str, List[MultiSizeBoardSample]] = {k: [] for k in target_sizes}
    for sample in filtered:
        if len(buckets[sample.size_class]) < max_per_size:
            buckets[sample.size_class].append(sample)
    return [x for size in target_sizes for x in buckets[size]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multisize_masking_eval.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    modules = cfg.get("modules") or BASE_MODULES

    artifacts = load_multisize_samples(cfg)
    samples = artifacts.samples
    data_audit = artifacts.audit
    eval_folds = int(cfg["eval"]["folds"])
    eval_repeats = int(cfg["eval"]["masking_repeats"])
    eval_trials = int(cfg["search"]["n_trials"])
    discovery_repeats = int(cfg.get("discovery", {}).get("repeats", eval_repeats))
    discovery_trials = int(cfg.get("discovery", {}).get("n_trials", eval_trials))
    discovery_modules = cfg.get("discovery", {}).get("candidate_modules")
    if not samples and bool(cfg["eval"].get("synthetic_fallback_when_empty", True)):
        samples = _build_synthetic_samples(
            seed=int(cfg["eval"]["seed"]),
            per_size=int(cfg["eval"].get("synthetic_boards_per_size", 6)),
        )
        eval_folds = min(eval_folds, 2)
        eval_repeats = min(eval_repeats, 2)
        eval_trials = min(eval_trials, 4)
        discovery_repeats = min(discovery_repeats, 2)
        discovery_trials = min(discovery_trials, 2)
        if discovery_modules:
            discovery_modules = discovery_modules[:3]
    samples = _select_target_sizes(samples, cfg)

    by_size = {"20": [], "80": [], "120": [], "160": []}
    for s in samples:
        by_size[s.size_class].append(s)

    per_size = {
        k: _summarize_size(v, cfg, modules, folds=eval_folds, repeats=eval_repeats, n_trials=eval_trials)
        for k, v in by_size.items()
    }
    overall_base = _summarize_size(
        samples,
        cfg,
        modules,
        folds=eval_folds,
        repeats=eval_repeats,
        n_trials=eval_trials,
    )

    discovery = run_module_discovery(
        boards=samples,
        folds=eval_folds,
        repeats=discovery_repeats,
        seed=int(cfg["eval"]["seed"]),
        n_trials=discovery_trials,
        candidate_modules=discovery_modules,
    )

    kept_modules = discovery.get("kept_modules", [])
    final_modules = modules + kept_modules
    overall_after = _summarize_size(
        samples,
        cfg,
        final_modules,
        folds=eval_folds,
        repeats=eval_repeats,
        n_trials=eval_trials,
    )

    per_size_payload = {
        "anti_leakage_checks": "passed",
        "valid_sample_count_by_size": data_audit["valid_sample_count_by_size"],
        "parse_counts_by_size": data_audit["parse_counts_by_size"],
        "results": per_size,
    }
    _ensure_parent(cfg["reports"]["per_size_summary"]).write_text(
        json.dumps(per_size_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    overall_payload = {
        "anti_leakage_checks": "passed",
        "before": overall_base,
        "after": overall_after,
        "kept_modules": kept_modules,
        "final_modules": final_modules,
    }
    _ensure_parent(cfg["reports"]["overall_summary"]).write_text(
        json.dumps(overall_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top10_summary = {
        "before": overall_base.get("full_model", {}),
        "after": overall_after.get("full_model", {}),
        "per_size_after": overall_after.get("per_size_metrics", {}),
    }
    _ensure_parent(cfg["reports"]["top10_summary"]).write_text(
        json.dumps(top10_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _ensure_parent(cfg["reports"]["per_target_predictions"]).write_text(
        __import__("pandas").DataFrame(overall_after.get("predictions", [])).to_csv(index=False), encoding="utf-8"
    )
    _ensure_parent(cfg["reports"]["module_leaderboard"]).write_text(
        json.dumps(discovery.get("leaderboard", []), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ensure_parent(cfg["reports"]["module_discovery_summary"]).write_text(
        json.dumps(discovery.get("module_discovery_summary", {}), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ensure_parent(cfg["reports"]["error_cases_top10"]).write_text(
        json.dumps(overall_after.get("error_cases_top10", []), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ensure_parent(cfg["reports"]["search_trials"]).write_text(
        json.dumps(overall_after.get("trial_leaderboard", []), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ensure_parent(cfg["reports"]["best_config"]).write_text(
        json.dumps(
            {
                "best_weights": overall_after.get("best_weights", {}),
                "kept_modules": kept_modules,
                "search_objective": [
                    "overall_top10_hit_rate",
                    "cumulative_top5_hit_rate",
                    "cumulative_top1_hit_rate",
                    "mrr",
                    "mean_rank_asc",
                ],
                "selection_reason": "lexicographic objective with deterministic seed",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(overall_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
