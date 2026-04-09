#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference_config import load_module_weights
from src.inference_service import run_inference


Cell = Tuple[int, int]
Board = List[List[int]]


@dataclass
class EvalCase:
    board_id: str
    full_board: Board


def _validate_full_board(board: Board) -> None:
    if not board or not board[0]:
        raise ValueError("board must be non-empty")
    cols = len(board[0])
    if any(len(row) != cols for row in board):
        raise ValueError("board must be rectangular")
    flat = [int(v) for row in board for v in row]
    n_total = len(flat)
    if len(set(flat)) != n_total:
        raise ValueError("board values must be unique")
    if sorted(flat) != list(range(1, n_total + 1)):
        raise ValueError("board values must be exactly 1..N")


def _iter_candidate_files(input_dir: Path) -> Iterable[Path]:
    for p in input_dir.rglob("*.json"):
        if "reports" in p.parts:
            continue
        yield p


def _extract_boards_from_json(path: Path) -> List[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]]
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "boards" in raw and isinstance(raw["boards"], list):
        records = raw["boards"]
    elif isinstance(raw, dict) and "grid" in raw:
        records = [raw]
    else:
        return []

    out: List[EvalCase] = []
    for i, rec in enumerate(records):
        grid = rec.get("full_board") or rec.get("grid")
        if not isinstance(grid, list):
            continue
        board = [[int(v) for v in row] for row in grid]
        _validate_full_board(board)
        out.append(EvalCase(board_id=str(rec.get("board_id", f"{path.stem}:{i}")), full_board=board))
    return out


def load_eval_cases(input_dir: Path) -> List[EvalCase]:
    if not input_dir.exists():
        raise ValueError(f"Fail-fast: input-dir does not exist: {input_dir}")

    out: List[EvalCase] = []
    errors: List[str] = []
    for p in _iter_candidate_files(input_dir):
        try:
            out.extend(_extract_boards_from_json(p))
        except Exception as exc:
            errors.append(f"{p}: {exc}")

    if not out:
        msg = "Fail-fast: no valid board json found under input-dir"
        if errors:
            msg += f"; examples={errors[:3]}"
        raise ValueError(msg)
    return out


def _masked_board(full_board: Board, masking_ratio: float, rnd: random.Random) -> Tuple[Board, List[Cell]]:
    rows, cols = len(full_board), len(full_board[0])
    n_total = rows * cols
    mask_count = int(math.floor(n_total * masking_ratio))
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rnd.shuffle(cells)
    masked = set(cells[:mask_count])
    board = [[-1 if (r, c) in masked else full_board[r][c] for c in range(cols)] for r in range(rows)]
    return board, list(masked)


def _rank_of_true(candidates: Sequence[Dict[str, Any]], true_cell: Cell) -> int:
    for idx, cand in enumerate(candidates, start=1):
        if (cand["row"] - 1, cand["col"] - 1) == true_cell:
            return idx
    return len(candidates) + 1


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    ranks = [int(r["rank"]) for r in rows]
    return {
        "num_cases": len(rows),
        "top1_hit_rate": sum(1 for x in ranks if x <= 1) / max(1, len(ranks)),
        "top3_hit_rate": sum(1 for x in ranks if x <= 3) / max(1, len(ranks)),
        "top5_hit_rate": sum(1 for x in ranks if x <= 5) / max(1, len(ranks)),
        "mrr": sum(1.0 / x for x in ranks) / max(1, len(ranks)),
        "mean_rank": sum(ranks) / max(1, len(ranks)),
        "median_rank": float(median(ranks)) if ranks else 0.0,
    }


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: v / total for k, v in w.items()}


def random_weight_search(
    modules: List[str],
    trials: int,
    seed: int,
    eval_cases: List[Tuple[EvalCase, Board, int, Cell]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    best: Dict[str, float] | None = None
    best_metric = -1.0
    records: List[Dict[str, Any]] = []
    for t in range(trials):
        raw = {m: rnd.random() + 1e-9 for m in modules}
        w = _normalize_weights(raw)
        case_rows = []
        for case, masked_board, target_number, true_cell in eval_cases:
            res = run_inference(masked_board, target_number, source="weight_search", module_weights=w)
            rank = _rank_of_true(res["candidate_cells"], true_cell)
            case_rows.append({"rank": rank})
        metric = _metrics(case_rows)
        rec = {"trial": t, **{f"w_{k}": v for k, v in w.items()}, **metric}
        records.append(rec)
        if metric["top3_hit_rate"] > best_metric:
            best_metric = metric["top3_hit_rate"]
            best = w
    assert best is not None
    return best, records


def _ablation(
    modules: List[str],
    best_weights: Dict[str, float],
    eval_cases: List[Tuple[EvalCase, Board, int, Cell]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full_metrics_rows = []
    for case, masked_board, target_number, true_cell in eval_cases:
        res = run_inference(masked_board, target_number, source="ablation_full", module_weights=best_weights)
        full_metrics_rows.append({"rank": _rank_of_true(res["candidate_cells"], true_cell)})
    rows.append({"variant": "full", **_metrics(full_metrics_rows)})

    for drop in modules:
        kept = [m for m in modules if m != drop]
        weights = _normalize_weights({m: best_weights[m] for m in kept})
        metric_rows = []
        for case, masked_board, target_number, true_cell in eval_cases:
            res = run_inference(masked_board, target_number, source=f"ablation_drop_{drop}", module_weights=weights)
            metric_rows.append({"rank": _rank_of_true(res["candidate_cells"], true_cell)})
        rows.append({"variant": f"drop_{drop}", **_metrics(metric_rows)})
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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
    parser.add_argument("--input-dir", default="gogo")
    parser.add_argument("--output-dir", default="reports/gogo_eval")
    parser.add_argument("--masking-ratio", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--weight-trials", type=int, default=80)
    args = parser.parse_args()

    cases = load_eval_cases(Path(args.input_dir))
    baseline_weights = load_module_weights()
    modules = list(baseline_weights.keys())

    instantiated: List[Tuple[EvalCase, Board, int, Cell]] = []
    for seed in args.seeds:
        rnd = random.Random(seed)
        for case in cases:
            masked_board, masked_cells = _masked_board(case.full_board, args.masking_ratio, rnd)
            for cell in masked_cells:
                instantiated.append((case, masked_board, case.full_board[cell[0]][cell[1]], cell))

    per_case_rows: List[Dict[str, Any]] = []
    for case, masked_board, target_number, true_cell in instantiated:
        res = run_inference(masked_board, target_number, source="gogo_formal_eval")
        rank = _rank_of_true(res["candidate_cells"], true_cell)
        best = res["best_cell"]
        per_case_rows.append(
            {
                "board_id": case.board_id,
                "target_number": target_number,
                "true_row": true_cell[0] + 1,
                "true_col": true_cell[1] + 1,
                "pred_row": best["row"],
                "pred_col": best["col"],
                "rank": rank,
                "top1_hit": int(rank <= 1),
                "top3_hit": int(rank <= 3),
                "top5_hit": int(rank <= 5),
                "score": best["score"],
                "confidence_1_to_100": best["confidence_1_to_100"],
            }
        )

    summary = _metrics(per_case_rows)

    best_weights, search_rows = random_weight_search(modules, args.weight_trials, args.seeds[0], instantiated)
    tuned_rows = []
    for case, masked_board, target_number, true_cell in instantiated:
        res = run_inference(masked_board, target_number, source="gogo_formal_eval_tuned", module_weights=best_weights)
        tuned_rows.append({"rank": _rank_of_true(res["candidate_cells"], true_cell)})
    tuned_summary = _metrics(tuned_rows)
    ablation_rows = _ablation(modules, best_weights, instantiated)

    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "per_case_results.csv", per_case_rows)
    _write_csv(out_dir / "weight_search_results.csv", search_rows)
    _write_csv(out_dir / "ablation_results.csv", ablation_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_dir": args.input_dir,
                "masking_ratio": args.masking_ratio,
                "seeds": args.seeds,
                "baseline_weights": baseline_weights,
                "best_weights": best_weights,
                "baseline_metrics": summary,
                "best_metrics": tuned_summary,
                "delta": {
                    "top1_hit_rate": tuned_summary["top1_hit_rate"] - summary["top1_hit_rate"],
                    "top3_hit_rate": tuned_summary["top3_hit_rate"] - summary["top3_hit_rate"],
                    "top5_hit_rate": tuned_summary["top5_hit_rate"] - summary["top5_hit_rate"],
                    "mrr": tuned_summary["mrr"] - summary["mrr"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
