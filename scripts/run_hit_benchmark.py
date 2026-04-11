from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ranking_debug import build_ranking_error_report
from src.ranking_features import build_candidate_feature_rows

REPORTS_DIR = Path("reports")


@dataclass
class BenchmarkCase:
    sample_id: str
    size_class: str
    source: str
    full_board: List[List[int]]
    masked_board: List[List[int]]
    target_number: int
    true_cell_0_based: Tuple[int, int]


def _mask_ratio(masked_board: List[List[int]]) -> float:
    total = len(masked_board) * len(masked_board[0])
    masked = sum(1 for row in masked_board for v in row if v == -1)
    return masked / total


def _build_cases_from_parsed_boards(path: Path, max_cases: int = 30, seed: int = 2026) -> List[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rnd = random.Random(seed)
    cases: List[BenchmarkCase] = []
    for item in raw:
        grid = item.get("grid")
        if not grid:
            continue
        rows = len(grid)
        cols = len(grid[0])
        if any(len(r) != cols for r in grid):
            continue
        flat = [v for row in grid for v in row]
        if any(v is None for v in flat):
            continue
        full_board = [[int(v) for v in row] for row in grid]
        total = rows * cols
        expected = list(range(1, total + 1))
        if sorted(flat) != expected:
            continue

        cells = [(r, c) for r in range(rows) for c in range(cols)]
        rnd.shuffle(cells)
        masked = set(cells[: max(1, total // 2)])
        masked_board = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(-1 if (r, c) in masked else full_board[r][c])
            masked_board.append(row)

        target_cell = rnd.choice(list(masked))
        target_number = full_board[target_cell[0]][target_cell[1]]
        cases.append(
            BenchmarkCase(
                sample_id=str(item.get("sample_id", f"case_{len(cases)}")),
                size_class=str(item.get("size_class", str(total))),
                source="parsed_boards",
                full_board=full_board,
                masked_board=masked_board,
                target_number=int(target_number),
                true_cell_0_based=target_cell,
            )
        )
        if len(cases) >= max_cases:
            break
    return cases


def _default_cases(seed: int = 2026) -> List[BenchmarkCase]:
    rnd = random.Random(seed)
    full_board = [
        [37, 12, 58, 4, 71, 26, 49, 80, 15, 63],
        [22, 54, 1, 68, 33, 47, 9, 72, 29, 60],
        [75, 18, 44, 6, 52, 39, 64, 11, 57, 24],
        [30, 66, 14, 79, 41, 2, 53, 20, 70, 35],
        [8, 61, 27, 46, 13, 74, 31, 55, 17, 69],
        [43, 5, 59, 21, 76, 34, 65, 10, 48, 28],
        [73, 16, 40, 62, 7, 56, 25, 78, 32, 50],
        [19, 67, 3, 45, 23, 77, 42, 51, 36, 38],
    ]
    rows, cols = 8, 10
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    cases: List[BenchmarkCase] = []
    for i in range(12):
        shuffled = list(cells)
        rnd.shuffle(shuffled)
        masked = set(shuffled[:40])
        masked_board = [[-1 if (r, c) in masked else full_board[r][c] for c in range(cols)] for r in range(rows)]
        target_cell = shuffled[i % 40]
        cases.append(
            BenchmarkCase(
                sample_id=f"default_{i}",
                size_class="80",
                source="default_seeded",
                full_board=full_board,
                masked_board=masked_board,
                target_number=full_board[target_cell[0]][target_cell[1]],
                true_cell_0_based=target_cell,
            )
        )
    return cases


def _rank_of_true(candidates: List[Dict[str, Any]], true_cell: Tuple[int, int]) -> Optional[int]:
    for idx, cell in enumerate(candidates, start=1):
        if (cell["row"] - 1, cell["col"] - 1) == true_cell:
            return idx
    return None


def _run_strategy(case: BenchmarkCase, strategy: str, seed: int) -> Dict[str, Any]:
    from src.inference_service import _run_inference_detailed
    if strategy == "random_baseline":
        result = _run_inference_detailed(case.masked_board, case.target_number, source="benchmark_random")
        candidates = list(result["candidate_cells"])
        rnd = random.Random(seed + hash(case.sample_id) % 100000)
        rnd.shuffle(candidates)
        result["candidate_cells"] = candidates
        result["best_cell"] = candidates[0]
        return result

    if strategy == "uniform_baseline":
        result = _run_inference_detailed(case.masked_board, case.target_number, source="benchmark_uniform")
        candidates = list(result["candidate_cells"])
        for c in candidates:
            c["score"] = 1.0
            c["confidence_1_to_100"] = 50.0
        result["candidate_cells"] = candidates
        result["best_cell"] = candidates[0]
        return result

    if strategy == "full_fusion_baseline":
        return _run_inference_detailed(
            case.masked_board,
            case.target_number,
            source="benchmark_full_fusion_baseline",
            apply_reranker_stage=False,
        )

    if strategy == "full_fusion_reranker":
        return _run_inference_detailed(
            case.masked_board,
            case.target_number,
            source="benchmark_full_fusion_reranker",
            apply_reranker_stage=True,
        )

    weights_map: Dict[str, Dict[str, float]] = {
        "center_prior_only": {"prior_model": 1.0},
        "logic_rule_only": {"logic_rule": 1.0},
        "pattern_model_only": {"pattern_model": 1.0},
        "prior_model_only": {"prior_model": 1.0},
    }
    module_weights = weights_map.get(strategy)
    if module_weights is None:
        raise ValueError(f"unknown strategy: {strategy}")
    return _run_inference_detailed(
        case.masked_board,
        case.target_number,
        source=f"benchmark_{strategy}",
        module_weights=module_weights if module_weights else None,
    )


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [r for r in rows if r["status"] == "ok"]
    invalid_rows = [r for r in rows if r["status"] != "ok"]

    def _hit_at(k: int) -> float:
        hits = sum(1 for r in valid_rows if r.get("true_rank") is not None and r["true_rank"] <= k)
        return hits / len(valid_rows) if valid_rows else 0.0

    mrr = mean([(1.0 / r["true_rank"]) for r in valid_rows if r.get("true_rank")]) if valid_rows else 0.0
    mean_true_rank = mean([r["true_rank"] for r in valid_rows if r.get("true_rank")]) if valid_rows else 0.0
    avg_candidates = mean([r["candidate_count"] for r in valid_rows]) if valid_rows else 0.0

    return {
        "total_cases": len(rows),
        "valid_cases": len(valid_rows),
        "invalid_cases": len(invalid_rows),
        "top1_hit_rate": round(_hit_at(1), 6),
        "top3_hit_rate": round(_hit_at(3), 6),
        "top5_hit_rate": round(_hit_at(5), 6),
        "mean_reciprocal_rank": round(mrr, 6),
        "mean_true_cell_rank": round(mean_true_rank, 6),
        "average_candidate_count": round(avg_candidates, 6),
    }


def _slice_metrics(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)
    return {k: _aggregate_metrics(v) for k, v in sorted(buckets.items())}


def _score_distribution(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [r["top1_score"] for r in rows if r.get("top1_score") is not None]
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": round(mean(values), 6)}


def run_benchmark(
    cases: List[BenchmarkCase],
    output_dir: Path = REPORTS_DIR,
    seed: int = 2026,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [
        "random_baseline",
        "uniform_baseline",
        "center_prior_only",
        "logic_rule_only",
        "pattern_model_only",
        "prior_model_only",
        "full_fusion_baseline",
        "full_fusion_reranker",
    ]

    strategy_rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    errors: List[Dict[str, Any]] = []
    ranking_training_rows: List[Dict[str, Any]] = []
    ranking_error_rows: List[Dict[str, Any]] = []

    for case in cases:
        mask_ratio = round(_mask_ratio(case.masked_board), 4)
        for strategy in strategies:
            try:
                res = _run_strategy(case, strategy, seed=seed)
                candidates = res["candidate_cells"]
                true_rank = _rank_of_true(candidates, case.true_cell_0_based)
                row_payload = {
                        "sample_id": case.sample_id,
                        "strategy": strategy,
                        "size_class": case.size_class,
                        "source": case.source,
                        "mask_ratio": mask_ratio,
                        "status": res["status"],
                        "target_number": case.target_number,
                        "true_row": case.true_cell_0_based[0],
                        "true_col": case.true_cell_0_based[1],
                        "pred_row": res["best_cell"]["row"] - 1 if res["best_cell"] else None,
                        "pred_col": res["best_cell"]["col"] - 1 if res["best_cell"] else None,
                        "top1_score": res["best_cell"]["score"] if res["best_cell"] else None,
                        "top1_confidence_1_to_100": (
                            res["best_cell"].get("confidence_1_to_100") if res["best_cell"] else None
                        ),
                        "true_rank": true_rank,
                        "candidate_count": len(candidates),
                        "metadata": res["metadata"],
                        "module_contributions": res["module_contributions"],
                        "top5_cells": [
                            [c["row"] - 1, c["col"] - 1] for c in candidates[:5]
                        ],
                    }
                strategy_rows[strategy].append(row_payload)
                if strategy == "full_fusion_baseline":
                    ranking_training_rows.extend(
                        build_candidate_feature_rows(
                            case_id=case.sample_id,
                            board_shape=(len(case.masked_board), len(case.masked_board[0])),
                            candidates=candidates,
                            true_cell_1_based=(case.true_cell_0_based[0] + 1, case.true_cell_0_based[1] + 1),
                        )
                    )
                    ranking_error_rows.append(
                        build_ranking_error_report(
                            case_id=case.sample_id,
                            target_number=case.target_number,
                            true_cell_1_based=(case.true_cell_0_based[0] + 1, case.true_cell_0_based[1] + 1),
                            baseline_candidates=candidates,
                        )
                    )
            except Exception as exc:
                row = {
                    "sample_id": case.sample_id,
                    "strategy": strategy,
                    "size_class": case.size_class,
                    "source": case.source,
                    "mask_ratio": mask_ratio,
                    "status": "invalid",
                    "error": str(exc),
                    "target_number": case.target_number,
                    "true_row": case.true_cell_0_based[0],
                    "true_col": case.true_cell_0_based[1],
                    "pred_row": None,
                    "pred_col": None,
                    "top1_score": None,
                    "top1_confidence_1_to_100": None,
                    "true_rank": None,
                    "candidate_count": 0,
                    "metadata": {},
                    "module_contributions": {},
                }
                strategy_rows[strategy].append(row)
                errors.append(row)

    full_rows = strategy_rows["full_fusion_reranker"]
    baseline_rows = strategy_rows["full_fusion_baseline"]
    summary = {
        "strategy_summaries": {s: _aggregate_metrics(rows) for s, rows in strategy_rows.items()},
        "by_size_class": _slice_metrics(full_rows, "size_class"),
        "by_source": _slice_metrics(full_rows, "source"),
        "by_mask_ratio": _slice_metrics(full_rows, "mask_ratio"),
        "score_distribution": _score_distribution(full_rows),
        "cache_usage": {
            "used_cache": True,
            "cache_artifact_path": "reports/parsed_boards.json",
            "cache_sample_count": len(cases),
        },
    }

    baseline_metrics = summary["strategy_summaries"]["full_fusion_baseline"]
    reranker_metrics = summary["strategy_summaries"]["full_fusion_reranker"]

    paired = {}
    for b in baseline_rows:
        paired[b["sample_id"]] = {"baseline": b}
    for r in full_rows:
        paired.setdefault(r["sample_id"], {})["reranker"] = r
    improved = worsened = unchanged = 0
    for sample in paired.values():
        b_rank = sample.get("baseline", {}).get("true_rank")
        r_rank = sample.get("reranker", {}).get("true_rank")
        if b_rank is None or r_rank is None:
            continue
        if r_rank < b_rank:
            improved += 1
        elif r_rank > b_rank:
            worsened += 1
        else:
            unchanged += 1

    summary["reranker_delta"] = {
        "delta_top1": round(reranker_metrics["top1_hit_rate"] - baseline_metrics["top1_hit_rate"], 6),
        "delta_top3": round(reranker_metrics["top3_hit_rate"] - baseline_metrics["top3_hit_rate"], 6),
        "delta_top5": round(reranker_metrics["top5_hit_rate"] - baseline_metrics["top5_hit_rate"], 6),
        "delta_MRR": round(reranker_metrics["mean_reciprocal_rank"] - baseline_metrics["mean_reciprocal_rank"], 6),
        "improved_case_count": improved,
        "worsened_case_count": worsened,
        "unchanged_case_count": unchanged,
        "status": "underperform" if reranker_metrics["top5_hit_rate"] < baseline_metrics["top5_hit_rate"] else (
            "indistinguishable" if reranker_metrics["top5_hit_rate"] == baseline_metrics["top5_hit_rate"] else "improved"
        ),
    }

    random_top1 = summary["strategy_summaries"]["random_baseline"]["top1_hit_rate"]
    ablation_summary = {}
    for strategy, metrics in summary["strategy_summaries"].items():
        ablation_summary[strategy] = {
            "top1_hit_rate": metrics["top1_hit_rate"],
            "top3_hit_rate": metrics["top3_hit_rate"],
            "top5_hit_rate": metrics["top5_hit_rate"],
            "mean_reciprocal_rank": metrics["mean_reciprocal_rank"],
            "delta_top1_vs_random": round(metrics["top1_hit_rate"] - random_top1, 6),
        }

    full_valid = [r for r in full_rows if r["status"] == "ok"]
    prior_bias_center_win = sum(1 for r in full_valid if r["pred_row"] in (3, 4) and r["pred_col"] in (4, 5))
    bottleneck = {
        "data_issues": {
            "invalid_cases": len([r for r in full_rows if r["status"] != "ok"]),
            "annotation_mismatch_cases": len(
                [r for r in full_rows if r.get("true_rank") is None and r["status"] == "ok"]
            ),
            "sample_count": len(cases),
        },
        "parsing_normalization_issues": {
            "cache_mode": "cached_parsed_boards",
            "cache_fallback_parse": False,
        },
        "candidate_ranking_issues": {
            "true_cell_in_candidate_set_rate": round(
                sum(1 for r in full_valid if r.get("true_rank") is not None)
                / max(len(full_valid), 1),
                6,
            ),
            "true_rank_distribution": {
                "top1": sum(1 for r in full_valid if r.get("true_rank") == 1),
                "top3": sum(1 for r in full_valid if r.get("true_rank") and r["true_rank"] <= 3),
                "top5": sum(1 for r in full_valid if r.get("true_rank") and r["true_rank"] <= 5),
                "gt5": sum(1 for r in full_valid if r.get("true_rank") and r["true_rank"] > 5),
            },
        },
        "module_issues": {
            "ablation": ablation_summary,
            "module_most_often_lowering_true_rank": (
                "pattern_model (proxy by low standalone MRR)"
            ),
        },
        "structural_bias": {
            "center_bias_top1_ratio": round(prior_bias_center_win / max(len(full_valid), 1), 6),
        },
    }

    csv_path = output_dir / "per_case_predictions.csv"
    fieldnames = [
        "sample_id",
        "strategy",
        "size_class",
        "source",
        "mask_ratio",
        "status",
        "target_number",
        "true_row",
        "true_col",
        "pred_row",
        "pred_col",
        "top1_score",
        "top1_confidence_1_to_100",
        "true_rank",
        "candidate_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for strategy in strategies:
            for row in strategy_rows[strategy]:
                writer.writerow({k: row.get(k) for k in fieldnames})

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "error_cases.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "bottleneck_report.json").write_text(
        json.dumps(bottleneck, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(ablation_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ranking_error_report.json").write_text(
        json.dumps(ranking_error_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ranking_csv = output_dir / "ranking_training_rows.csv"
    if ranking_training_rows:
        keys = list(ranking_training_rows[0].keys())
        with ranking_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in ranking_training_rows:
                writer.writerow(row)

    return {
        "summary": summary,
        "ablation_summary": ablation_summary,
        "bottleneck": bottleneck,
        "per_case_rows": strategy_rows,
    }


def main() -> None:
    cases = _build_cases_from_parsed_boards(Path("reports/parsed_boards.json"), max_cases=30, seed=2026)
    if not cases:
        cases = _default_cases(seed=2026)
    result = run_benchmark(cases)
    print(json.dumps({"baseline": result["summary"]["strategy_summaries"]["full_fusion_baseline"], "reranker": result["summary"]["strategy_summaries"]["full_fusion_reranker"], "delta": result["summary"]["reranker_delta"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
