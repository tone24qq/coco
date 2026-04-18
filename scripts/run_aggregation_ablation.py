from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.inference_config import load_aggregator_config
from src.inference_service import _run_inference_detailed


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_pos(grid: List[List[int]], target_number: int) -> Tuple[int, int]:
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if int(v) == int(target_number):
                return r, c
    raise ValueError(f"target_number {target_number} not found")


def _mask_board(grid: List[List[int]], ratio: float, seed: int) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0])
    total = rows * cols
    mask_count = max(1, int(round(total * ratio)))
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    masked = [row[:] for row in grid]
    for idx in indices[:mask_count]:
        r, c = divmod(idx, cols)
        masked[r][c] = -1
    return masked


def _build_schemes(base_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    schemes: Dict[str, Dict[str, Any]] = {}

    def make(**kwargs: Any) -> Dict[str, Any]:
        cfg = deepcopy(base_cfg)
        cfg.update(kwargs)
        return cfg

    schemes["baseline"] = make(
        contribution_mode="weighted_sum",
        confidence_gate_threshold=0.5,
        low_confidence_weight_multiplier=1.0,
        use_centered_score=False,
        abstain_below_threshold=False,
    )
    schemes["hard_gate_050"] = make(
        contribution_mode="weighted_sum",
        confidence_gate_threshold=0.50,
        low_confidence_weight_multiplier=1.0,
        use_centered_score=False,
        abstain_below_threshold=True,
    )
    schemes["hard_gate_055"] = make(
        contribution_mode="weighted_sum",
        confidence_gate_threshold=0.55,
        low_confidence_weight_multiplier=1.0,
        use_centered_score=False,
        abstain_below_threshold=True,
    )
    schemes["hard_gate_060"] = make(
        contribution_mode="weighted_sum",
        confidence_gate_threshold=0.60,
        low_confidence_weight_multiplier=1.0,
        use_centered_score=False,
        abstain_below_threshold=True,
    )
    schemes["soft_gate_050"] = make(
        contribution_mode="weighted_sum",
        confidence_gate_threshold=0.50,
        low_confidence_weight_multiplier=0.2,
        use_centered_score=False,
        abstain_below_threshold=False,
    )
    schemes["centered_weighted_sum"] = make(
        contribution_mode="centered_weighted_sum",
        confidence_gate_threshold=0.5,
        low_confidence_weight_multiplier=1.0,
        use_centered_score=True,
        abstain_below_threshold=False,
    )
    return schemes


def _metric_from_ranks(ranks: List[int]) -> Dict[str, float]:
    if not ranks:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "mean_rank": 0.0, "mrr": 0.0}
    n = len(ranks)
    return {
        "top1": sum(1 for r in ranks if r <= 1) / n,
        "top3": sum(1 for r in ranks if r <= 3) / n,
        "top5": sum(1 for r in ranks if r <= 5) / n,
        "top10": sum(1 for r in ranks if r <= 10) / n,
        "mean_rank": sum(ranks) / n,
        "mrr": sum(1.0 / r for r in ranks) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-corpus", default="data/full_boards/holdout_real_80.jsonl")
    parser.add_argument("--mask-ratios", default="0.3,0.5")
    parser.add_argument("--masks-per-ratio", type=int, default=1)
    parser.add_argument("--max-targets-per-board", type=int, default=0)
    parser.add_argument("--output-summary", default="reports/aggregation_ablation_summary.json")
    parser.add_argument("--output-per-case", default="reports/aggregation_ablation_per_case.csv")
    parser.add_argument("--output-best", default="reports/aggregation_best_config.json")
    args = parser.parse_args()

    boards = _read_jsonl(Path(args.holdout_corpus))
    base_cfg = load_aggregator_config()
    schemes = _build_schemes(base_cfg)
    ratios = [float(x.strip()) for x in args.mask_ratios.split(",") if x.strip()]

    per_case: List[Dict[str, Any]] = []
    for board_row in boards:
        full_grid = board_row["grid"]
        rows = int(board_row["rows"])
        cols = int(board_row["cols"])
        board_id = str(board_row.get("board_id", "unknown"))
        targets = list(range(1, rows * cols + 1))
        if args.max_targets_per_board > 0:
            targets = targets[: args.max_targets_per_board]

        for ratio in ratios:
            for mask_idx in range(args.masks_per_ratio):
                seed = abs(hash((board_id, ratio, mask_idx))) % (10**9)
                masked_board = _mask_board(full_grid, ratio, seed)
                for target_number in targets:
                    true_r, true_c = _find_pos(full_grid, target_number)
                    if masked_board[true_r][true_c] != -1:
                        continue
                    for scheme_name, cfg in schemes.items():
                        out = _run_inference_detailed(
                            board=masked_board,
                            target_number=target_number,
                            source=f"ablation::{scheme_name}",
                            apply_reranker_stage=False,
                            aggregator_config=cfg,
                        )
                        ranked = out["candidate_cells"]
                        rank = None
                        for i, cand in enumerate(ranked, start=1):
                            if int(cand["row"]) - 1 == true_r and int(cand["col"]) - 1 == true_c:
                                rank = i
                                break
                        if rank is None:
                            continue
                        metadata = out.get("metadata", {})
                        per_case.append(
                            {
                                "scheme": scheme_name,
                                "board_id": board_id,
                                "size_class": f"{rows}x{cols}",
                                "mask_ratio": ratio,
                                "mask_idx": mask_idx,
                                "target_number": target_number,
                                "rank": rank,
                                "top1": int(rank <= 1),
                                "top3": int(rank <= 3),
                                "top5": int(rank <= 5),
                                "top10": int(rank <= 10),
                                "active_module_count_mean": float(metadata.get("active_module_count_mean", 0.0)),
                                "abstain_rate": float(metadata.get("abstain_rate", 0.0)),
                                "no_informative_modules_rate": float(metadata.get("no_informative_modules_rate", 0.0)),
                                "score_std": float(metadata.get("score_std", 0.0)),
                                "final_score_std": float(metadata.get("final_score_std", 0.0)),
                            }
                        )

    if not per_case:
        raise ValueError("no ablation cases executed")

    per_case_df = pd.DataFrame(per_case)
    per_case_path = Path(args.output_per_case)
    per_case_path.parent.mkdir(parents=True, exist_ok=True)
    per_case_df.to_csv(per_case_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    baseline_metrics: Dict[str, float] | None = None
    for scheme, sub in per_case_df.groupby("scheme"):
        ranks = sub["rank"].astype(int).tolist()
        m = _metric_from_ranks(ranks)
        row = {
            "scheme": scheme,
            **m,
            "case_count": int(len(sub)),
            "active_module_count_mean": float(sub["active_module_count_mean"].mean()),
            "abstain_rate": float(sub["abstain_rate"].mean()),
            "no_informative_modules_rate": float(sub["no_informative_modules_rate"].mean()),
            "score_std": float(sub["score_std"].mean()),
            "final_score_std": float(sub["final_score_std"].mean()),
        }
        if scheme == "baseline":
            baseline_metrics = row
        summary_rows.append(row)

    if baseline_metrics is None:
        raise ValueError("baseline scheme missing")

    for row in summary_rows:
        row["delta_top1"] = float(row["top1"] - baseline_metrics["top1"])
        row["delta_top3"] = float(row["top3"] - baseline_metrics["top3"])
        row["delta_top5"] = float(row["top5"] - baseline_metrics["top5"])
        row["delta_top10"] = float(row["top10"] - baseline_metrics["top10"])
        row["delta_mrr"] = float(row["mrr"] - baseline_metrics["mrr"])

    summary_rows.sort(key=lambda x: (x["top1"], x["top3"], x["top5"], x["mrr"]), reverse=True)
    recommended = summary_rows[0]

    summary_out = {
        "cases": int(len(per_case_df)),
        "schemes": summary_rows,
        "baseline": baseline_metrics,
        "recommended_default": recommended["scheme"],
        "decision_rule": "prioritize top1/top3/top5/mrr; top10 is secondary",
    }
    summary_path = Path(args.output_summary)
    summary_path.write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    best_path = Path(args.output_best)
    best_path.write_text(json.dumps(recommended, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"cases": len(per_case_df), "recommended_default": recommended["scheme"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
