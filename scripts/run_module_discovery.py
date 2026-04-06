#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.masking_eval.data_loader import load_full_boards
from src.masking_eval.discovery import run_module_discovery, write_discovery_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/masking_eval.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    boards, _ = load_full_boards(Path(cfg["data"]["path"]))

    repeats = int(cfg.get("discovery", {}).get("repeats", cfg["eval"]["masking_repeats"]))
    trials = int(cfg.get("discovery", {}).get("n_trials", cfg["search"]["n_trials"]))
    result = run_module_discovery(
        boards=boards,
        folds=int(cfg["eval"]["folds"]),
        repeats=repeats,
        seed=int(cfg["eval"]["seed"]),
        n_trials=trials,
        candidate_modules=cfg.get("discovery", {}).get("candidate_modules"),
    )
    write_discovery_outputs(
        result,
        leaderboard_path=Path("reports/module_leaderboard.json"),
        champion_path=Path("reports/champion_bundle.json"),
        summary_path=Path("reports/module_discovery_summary.json"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
