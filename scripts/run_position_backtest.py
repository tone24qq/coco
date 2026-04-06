#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from src.data_loader import discover_data_files, load_and_validate, write_data_audit
from src.eval import (
    ablation,
    center_baseline,
    density_baseline,
    evaluate_samples,
    random_baseline,
    tune_weights,
    write_case_predictions,
    write_json,
)


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/position_eval.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path(args.config))

    repo_root = Path(".").resolve()
    discovered = discover_data_files(repo_root)
    logging.info("discovered_data_files=%s", [str(p) for p in discovered])

    data_path = Path(cfg["data"]["path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    samples, audit = load_and_validate(data_path)
    write_data_audit(audit, Path(cfg["reports"]["data_audit"]))

    modules = [m for m, enabled in cfg["modules"].items() if enabled]
    if not modules:
        raise ValueError("No module enabled")

    grid_values = cfg["search"]["weight_grid"]
    best_weights = tune_weights(samples, modules, grid_values, k=cfg["eval"]["folds"])

    full_metrics, case_results = evaluate_samples(samples, best_weights, modules)
    random_metrics = random_baseline(samples, repeats=cfg["eval"]["random_repeats"], seed=cfg["eval"]["seed"])
    center_metrics = center_baseline(samples)
    density_metrics = density_baseline(samples)
    ablation_metrics = ablation(samples, best_weights, modules)

    summary = {
        "full_model": full_metrics,
        "random_baseline": random_metrics,
        "center_baseline": center_metrics,
        "density_baseline": density_metrics,
        "ablation": ablation_metrics,
        "enabled_modules": modules,
    }

    write_json(Path(cfg["reports"]["summary_report"]), summary)
    write_case_predictions(Path(cfg["reports"]["per_case_predictions"]), case_results)
    write_json(Path(cfg["reports"]["best_config"]), {"weights": best_weights, "modules": modules})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
