from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_aggregator_config
from src.inference_service import _run_inference_detailed

Board = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class EvalCase:
    size_class: str
    board_id: str
    board: Board
    target_cell: Cell
    target_number: int
    masked_board: Board


def _parse_sizes(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        rows, cols = token.split("x")
        out.append((int(rows), int(cols)))
    if not out:
        raise ValueError("at least one size class is required")
    return out


def _generate_full_board(rows: int, cols: int, rng: np.random.Generator) -> Board:
    vals = np.arange(1, rows * cols + 1, dtype=int)
    rng.shuffle(vals)
    return vals.reshape(rows, cols).tolist()


def _mask_50pct(board: Board, rng: np.random.Generator) -> Tuple[Board, List[Cell]]:
    rows = len(board)
    cols = len(board[0])
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    masked_cells = cells[: len(cells) // 2]
    masked_set = set(masked_cells)
    masked = [[-1 if (r, c) in masked_set else int(board[r][c]) for c in range(cols)] for r in range(rows)]
    return masked, masked_cells


def _iter_eval_cases(
    sizes: Sequence[Tuple[int, int]],
    boards_per_size: int,
    seed: int,
) -> Iterable[EvalCase]:
    rng = np.random.default_rng(seed)
    for rows, cols in sizes:
        for bid in range(boards_per_size):
            board = _generate_full_board(rows, cols, rng)
            masked, targets = _mask_50pct(board, rng)
            for cell in targets:
                yield EvalCase(
                    size_class=f"{rows}x{cols}",
                    board_id=f"{rows}x{cols}_{bid}",
                    board=board,
                    target_cell=cell,
                    target_number=int(board[cell[0]][cell[1]]),
                    masked_board=masked,
                )


def _rank_of_true(candidates: List[Dict[str, Any]], true_cell: Cell) -> int:
    for idx, cand in enumerate(candidates, start=1):
        if (int(cand["row"]) - 1, int(cand["col"]) - 1) == true_cell:
            return idx
    raise ValueError("true cell missing in candidate set")


def _topk_cluster_stats(candidates: List[Dict[str, Any]], k: int = 10) -> Dict[str, float]:
    top = candidates[: min(k, len(candidates))]
    cells = [(int(c["row"]) - 1, int(c["col"]) - 1) for c in top]
    if not cells:
        return {"topk_max_cluster_size": 0.0, "topk_adjacent_pair_rate": 0.0}
    visited = set()
    max_cluster = 0
    adjacent_pairs = 0
    total_pairs = 0
    for i, a in enumerate(cells):
        for j in range(i + 1, len(cells)):
            b = cells[j]
            total_pairs += 1
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                adjacent_pairs += 1
    for cell in cells:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in cells:
                if nxt in visited:
                    continue
                if max(abs(cur[0] - nxt[0]), abs(cur[1] - nxt[1])) <= 1:
                    visited.add(nxt)
                    stack.append(nxt)
        max_cluster = max(max_cluster, size)
    return {
        "topk_max_cluster_size": float(max_cluster),
        "topk_adjacent_pair_rate": float(adjacent_pairs / max(total_pairs, 1)),
    }


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ranks = [int(r["rank"]) for r in rows]
    if not ranks:
        return {}
    return {
        "num_targets": float(len(ranks)),
        "top1_hit_rate": float(sum(1 for r in ranks if r <= 1) / len(ranks)),
        "top3_hit_rate": float(sum(1 for r in ranks if r <= 3) / len(ranks)),
        "top5_hit_rate": float(sum(1 for r in ranks if r <= 5) / len(ranks)),
        "top10_hit_rate": float(sum(1 for r in ranks if r <= 10) / len(ranks)),
        "mean_rank": float(mean(ranks)),
        "median_rank": float(median(ranks)),
        "mrr": float(mean(1.0 / r for r in ranks)),
        "avg_top10_max_cluster_size": float(mean(float(r["topk_max_cluster_size"]) for r in rows)),
        "avg_top10_adjacent_pair_rate": float(mean(float(r["topk_adjacent_pair_rate"]) for r in rows)),
    }


def _variant_configs(base: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    def _with_spatial(**kwargs: Any) -> Dict[str, Any]:
        cfg = dict(base)
        spatial = dict(base.get("spatial_postprocess", {}))
        spatial.update(kwargs)
        cfg["spatial_postprocess"] = spatial
        return cfg

    return {
        "baseline": _with_spatial(enabled=False, method="spatial_penalty"),
        "A_spatial_hybrid": _with_spatial(
            enabled=True,
            method="spatial_penalty",
            distance_metric="hybrid",
            top_m=5,
        ),
        "B_spatial_chebyshev": _with_spatial(
            enabled=True,
            method="spatial_penalty",
            distance_metric="chebyshev",
            top_m=8,
            score_gap_gate=0.08,
        ),
        "C_evidence_aware_mmr": _with_spatial(
            enabled=True,
            method="evidence_aware_mmr",
            distance_metric="hybrid",
            top_m=8,
            mmr_lambda=0.35,
            mmr_distance_scale=0.09,
            score_gap_gate=0.08,
        ),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="4x4,4x5,5x5,8x10,10x8,10x12")
    parser.add_argument("--boards-per-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--output-dir", default="reports/non_training_cluster_study")
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    cases = list(_iter_eval_cases(sizes=sizes, boards_per_size=int(args.boards_per_size), seed=int(args.seed)))
    if not cases:
        raise ValueError("no evaluation cases generated")

    base_agg = load_aggregator_config()
    variants = _variant_configs(base_agg)

    per_case_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    by_variant: Dict[str, List[Dict[str, Any]]] = {name: [] for name in variants}

    for variant_name, agg_cfg in variants.items():
        for case in cases:
            out = _run_inference_detailed(
                board=case.masked_board,
                target_number=case.target_number,
                source=f"cluster_study:{variant_name}",
                apply_reranker_stage=False,
                aggregator_config=agg_cfg,
            )
            cands = out["candidate_cells"]
            rank = _rank_of_true(cands, case.target_cell)
            cluster_stats = _topk_cluster_stats(cands, k=10)
            row = {
                "variant": variant_name,
                "size_class": case.size_class,
                "board_id": case.board_id,
                "target_row": case.target_cell[0] + 1,
                "target_col": case.target_cell[1] + 1,
                "target_number": case.target_number,
                "rank": rank,
                "top1_hit": int(rank <= 1),
                "top3_hit": int(rank <= 3),
                "top5_hit": int(rank <= 5),
                "top10_hit": int(rank <= 10),
                "spatial_applied": int(bool(out["metadata"].get("spatial_postprocess_applied", False))),
                "spatial_affected_count": int(out["metadata"].get("spatial_postprocess_affected_count", 0)),
                "spatial_total_penalty": float(out["metadata"].get("spatial_postprocess_total_penalty", 0.0)),
                **cluster_stats,
            }
            by_variant[variant_name].append(row)
            per_case_rows.append(row)

        metrics = _aggregate_metrics(by_variant[variant_name])
        summary_rows.append({"variant": variant_name, **metrics})

    baseline = next((r for r in summary_rows if r["variant"] == "baseline"), None)
    if baseline is None:
        raise ValueError("baseline summary missing")
    comparison_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        comparison_rows.append(
            {
                **row,
                "delta_top5_vs_baseline": float(row["top5_hit_rate"] - baseline["top5_hit_rate"]),
                "delta_top10_vs_baseline": float(row["top10_hit_rate"] - baseline["top10_hit_rate"]),
                "delta_mrr_vs_baseline": float(row["mrr"] - baseline["mrr"]),
                "delta_mean_rank_vs_baseline": float(row["mean_rank"] - baseline["mean_rank"]),
                "delta_cluster_size_vs_baseline": float(
                    row["avg_top10_max_cluster_size"] - baseline["avg_top10_max_cluster_size"]
                ),
            }
        )

    # prioritize top5/top10 improvement, then mrr, then cluster reduction and mean_rank drop
    ranked = sorted(
        comparison_rows,
        key=lambda r: (
            float(r["delta_top5_vs_baseline"]),
            float(r["delta_top10_vs_baseline"]),
            float(r["delta_mrr_vs_baseline"]),
            -float(r["delta_cluster_size_vs_baseline"]),
            -float(r["delta_mean_rank_vs_baseline"]),
        ),
        reverse=True,
    )
    best = ranked[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_case_results.csv", per_case_rows)
    write_csv(output_dir / "summary_comparison.csv", comparison_rows)
    report = {
        "dataset": {
            "sizes": [f"{r}x{c}" for r, c in sizes],
            "boards_per_size": int(args.boards_per_size),
            "num_cases": len(cases),
            "seed": int(args.seed),
            "masking_ratio": 0.5,
        },
        "variants": variants,
        "summary": comparison_rows,
        "best_variant": best,
        "recommended_for_mainline": bool(
            best["variant"] != "baseline"
            and best["delta_top5_vs_baseline"] > 0.0
            and best["delta_top10_vs_baseline"] >= 0.0
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
