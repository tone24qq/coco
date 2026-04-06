#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.masking_eval.backtest import run_backtest, write_outputs
from src.masking_eval.data_loader import discover_board_files, load_full_boards, write_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/masking_eval.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    discovered = discover_board_files(Path("."))
    logging.info("discovered_board_files=%s", [str(x) for x in discovered])

    boards, audit = load_full_boards(Path(cfg["data"]["path"]))
    write_audit(Path(cfg["reports"]["data_audit"]), audit)

    modules = [k for k, v in cfg["modules"].items() if v]
    result = run_backtest(
        boards=boards,
        folds=int(cfg["eval"]["folds"]),
        repeats=int(cfg["eval"]["masking_repeats"]),
        seed=int(cfg["eval"]["seed"]),
        modules=modules,
        n_trials=int(cfg["search"]["n_trials"]),
    )
    write_outputs(
        result,
        summary_path=Path(cfg["reports"]["summary_report"]),
        pred_path=Path(cfg["reports"]["per_target_predictions"]),
        config_path=Path(cfg["reports"]["best_config"]),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
