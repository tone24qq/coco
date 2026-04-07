#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.masking_eval.discovery import run_module_discovery, write_discovery_outputs  # noqa: E402
from src.multi_size_data_loader import load_multisize_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multisize_masking_eval.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    artifacts = load_multisize_samples(cfg)
    result = run_module_discovery(
        boards=artifacts.samples,
        folds=int(cfg["eval"]["folds"]),
        repeats=int(cfg.get("discovery", {}).get("repeats", cfg["eval"]["masking_repeats"])),
        seed=int(cfg["eval"]["seed"]),
        n_trials=int(cfg.get("discovery", {}).get("n_trials", cfg["search"]["n_trials"])),
        candidate_modules=cfg.get("discovery", {}).get("candidate_modules"),
    )
    write_discovery_outputs(
        result,
        leaderboard_path=Path(cfg["reports"]["module_leaderboard"]),
        champion_path=Path(cfg["reports"]["champion_bundle"]),
        summary_path=Path(cfg["reports"]["discovery_summary"]),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
