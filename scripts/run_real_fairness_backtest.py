from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_module_weights  # noqa: E402
from src.mainline_eval import (  # noqa: E402
    FullBoardRecord,
    normalize_weights,
    random_weight_candidates,
    run_weighted_eval,
)

Board = List[List[int]]


@dataclass
class FairnessDataset:
    boards: List[FullBoardRecord]
    sources: Dict[str, int]
    size_counts: Dict[str, int]


def _validate_full_board(board: Board) -> None:
    if not board or not board[0]:
        raise ValueError("board must be non-empty")
    rows = len(board)
    cols = len(board[0])
    if any(len(row) != cols for row in board):
        raise ValueError("board must be rectangular")
    vals = [int(v) for row in board for v in row]
    n = rows * cols
    if sorted(vals) != list(range(1, n + 1)):
        raise ValueError("board must be full permutation 1..N")


def _load_full_boards_10x8(path: Path) -> List[FullBoardRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[FullBoardRecord] = []
    for item in payload:
        board = item.get("grid")
        if not isinstance(board, list):
            continue
        _validate_full_board(board)
        out.append(
            FullBoardRecord(
                board_id=str(item.get("board_id", f"10x8:{len(out)}")),
                board=board,
                source=f"real_full_board:{path}",
            )
        )
    return out


def _reconstruct_from_competitive_cases(path: Path) -> List[FullBoardRecord]:
    by_shape: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            board = row["board"]
            shape = (len(board), len(board[0]))
            by_shape.setdefault(shape, []).append(row)
    out: List[FullBoardRecord] = []
    for shape, rows in sorted(by_shape.items()):
        n_rows, n_cols = shape
        merged: List[List[int | None]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        for item in rows:
            board = item["board"]
            for r in range(n_rows):
                for c in range(n_cols):
                    value = int(board[r][c])
                    if value == -1:
                        continue
                    prev = merged[r][c]
                    if prev is not None and prev != value:
                        raise ValueError(f"inconsistent competitive case merge at shape={shape} cell={(r, c)}")
                    merged[r][c] = value
            tr, tc = item["true_cell"]
            tr0, tc0 = int(tr) - 1, int(tc) - 1
            target = int(item["target_number"])
            prev = merged[tr0][tc0]
            if prev is not None and prev != target:
                raise ValueError(f"inconsistent target merge at shape={shape} cell={(tr0, tc0)}")
            merged[tr0][tc0] = target
        if any(v is None for row in merged for v in row):
            raise ValueError(f"shape={shape} cannot reconstruct complete full board from competitive cases")
        full = [[int(v) for v in row] for row in merged]
        _validate_full_board(full)
        out.append(
            FullBoardRecord(
                board_id=f"competitive_reconstructed_{n_rows}x{n_cols}",
                board=full,
                source=f"competitive_cases:{path}",
            )
        )
    return out


def load_real_fairness_dataset() -> FairnessDataset:
    boards: List[FullBoardRecord] = []
    boards.extend(_load_full_boards_10x8(Path("samples/data/full_boards_10x8.json")))
    boards.extend(_reconstruct_from_competitive_cases(Path("data/competitive_cases.jsonl")))
    if not boards:
        raise ValueError("no real full boards available")
    sources: Dict[str, int] = {}
    size_counts: Dict[str, int] = {}
    for rec in boards:
        sources[rec.source] = sources.get(rec.source, 0) + 1
        size = f"{len(rec.board)}x{len(rec.board[0])}"
        size_counts[size] = size_counts.get(size, 0) + 1
    return FairnessDataset(boards=boards, sources=sources, size_counts=size_counts)


def fairness_objective(metrics: Dict[str, float]) -> float:
    return (
        0.40 * float(metrics.get("top5_hit_rate", 0.0))
        + 0.20 * float(metrics.get("corner_top10_hit_rate", 0.0))
        + 0.20 * float(metrics.get("edge_top10_hit_rate", 0.0))
        + 0.10 * float(metrics.get("corner_top3_hit_rate", 0.0))
        + 0.10 * float(metrics.get("edge_top3_hit_rate", 0.0))
        - 0.10 * max(0.0, float(metrics.get("center_top10_candidate_share", 0.0)) - 0.55)
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_ablation(
    boards: Sequence[FullBoardRecord],
    improved_weights: Dict[str, float],
    masking_ratio: float,
    repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for drop in sorted(improved_weights.keys()):
        proposal = {k: v for k, v in improved_weights.items() if k != drop}
        proposal = normalize_weights(proposal)
        _, metrics = run_weighted_eval(
            boards=boards,
            weights=proposal,
            masking_ratio=masking_ratio,
            repeats=repeats,
            seed=seed,
            apply_reranker_stage=False,
        )
        rows.append(
            {
                "dropped_module": drop,
                "top5_hit_rate": metrics["top5_hit_rate"],
                "corner_top10_hit_rate": metrics["corner_top10_hit_rate"],
                "edge_top10_hit_rate": metrics["edge_top10_hit_rate"],
                "center_top10_candidate_share": metrics["center_top10_candidate_share"],
                "corner_mean_true_rank": metrics["corner_mean_true_rank"],
                "edge_mean_true_rank": metrics["edge_mean_true_rank"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports/real_fairness_backtest")
    parser.add_argument("--masking-ratio", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--weight-trials", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = load_real_fairness_dataset()
    write_json(
        output_dir / "dataset_summary.json",
        {
            "num_boards": len(dataset.boards),
            "source_breakdown": dataset.sources,
            "size_breakdown": dataset.size_counts,
            "boards": [asdict(x) for x in dataset.boards],
        },
    )
    baseline_weights = normalize_weights(load_module_weights())
    baseline_rows, baseline_metrics = run_weighted_eval(
        boards=dataset.boards,
        weights=baseline_weights,
        masking_ratio=float(args.masking_ratio),
        repeats=int(args.repeats),
        seed=int(args.seed),
        apply_reranker_stage=False,
    )

    best_weights = baseline_weights
    best_score = fairness_objective(baseline_metrics)
    search_rows: List[Dict[str, Any]] = []
    proposals = random_weight_candidates(
        list(baseline_weights.keys()),
        trials=int(args.weight_trials),
        seed=int(args.seed),
    )
    for idx, w in enumerate(proposals):
        _, metrics = run_weighted_eval(
            boards=dataset.boards,
            weights=w,
            masking_ratio=float(args.masking_ratio),
            repeats=int(args.repeats),
            seed=int(args.seed),
            apply_reranker_stage=False,
        )
        score = fairness_objective(metrics)
        search_rows.append(
            {
                "trial": idx,
                "objective": score,
                "top5_hit_rate": metrics["top5_hit_rate"],
                "corner_top10_hit_rate": metrics["corner_top10_hit_rate"],
                "edge_top10_hit_rate": metrics["edge_top10_hit_rate"],
                "center_top10_candidate_share": metrics["center_top10_candidate_share"],
            }
        )
        if score > best_score:
            best_score = score
            best_weights = w
    improved_rows, improved_metrics = run_weighted_eval(
        boards=dataset.boards,
        weights=best_weights,
        masking_ratio=float(args.masking_ratio),
        repeats=int(args.repeats),
        seed=int(args.seed),
        apply_reranker_stage=False,
    )

    delta = {
        key: float(improved_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0))
        for key in improved_metrics
        if isinstance(improved_metrics.get(key), (int, float)) and isinstance(baseline_metrics.get(key), (int, float))
    }
    ablation_rows = _run_ablation(
        boards=dataset.boards,
        improved_weights=best_weights,
        masking_ratio=float(args.masking_ratio),
        repeats=int(args.repeats),
        seed=int(args.seed),
    )

    min_zone_samples = min(
        int(improved_metrics.get("corner_sample_count", 0)),
        int(improved_metrics.get("edge_sample_count", 0)),
        int(improved_metrics.get("center_sample_count", 0)),
    )
    sufficient = min_zone_samples >= 30
    report = {
        "dataset": {
            "num_boards": len(dataset.boards),
            "size_breakdown": dataset.size_counts,
            "source_breakdown": dataset.sources,
        },
        "baseline_weights": baseline_weights,
        "improved_weights": best_weights,
        "baseline_metric_summary": baseline_metrics,
        "best_metric_summary": improved_metrics,
        "delta_vs_baseline": delta,
        "fairness_objective_baseline": fairness_objective(baseline_metrics),
        "fairness_objective_improved": fairness_objective(improved_metrics),
        "zone_sample_sufficiency": {
            "min_zone_samples": min_zone_samples,
            "threshold": 30,
            "is_sufficient": sufficient,
        },
        "claim_status": (
            "position_bias_significantly_improved"
            if (
                sufficient
                and delta.get("corner_top10_hit_rate", 0.0) > 0.02
                and delta.get("corner_mean_true_rank", 0.0) < 0.0
            )
            else "not_yet_proven"
        ),
        "claim_reason": (
            "real-board evidence shows corner hit-rate up and corner mean rank down with adequate zone samples"
            if (
                sufficient
                and delta.get("corner_top10_hit_rate", 0.0) > 0.02
                and delta.get("corner_mean_true_rank", 0.0) < 0.0
            )
            else "real-board evidence is still insufficient or improvement signal is not strong enough"
        ),
    }

    write_json(output_dir / "summary.json", report)
    write_csv(output_dir / "baseline_per_case.csv", baseline_rows)
    write_csv(output_dir / "improved_per_case.csv", improved_rows)
    write_csv(output_dir / "weight_search_trials.csv", search_rows)
    write_csv(output_dir / "ablation_results.csv", ablation_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
