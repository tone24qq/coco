from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.inference_service import _run_inference_detailed

Board = List[List[int]]
Cell = Tuple[int, int]

CORE_MODULES = [
    "logic_rule",
    "structural_consistency",
    "pattern_model",
    "focus_score",
    "connectivity_heatmap",
    "difference_trend",
    "skip_patterns",
    "mirror_sequences",
    "tail_analyzer",
    "neighborhood_association",
    "local_arithmetic_relation",
]


@dataclass
class FullBoardRecord:
    board_id: str
    board: Board
    source: str


@dataclass
class DiscoveryArtifacts:
    boards: List[FullBoardRecord]
    invalid_reasons: Dict[str, int]


def validate_full_board(full_board: Board) -> None:
    if not full_board or not full_board[0]:
        raise ValueError("board must be non-empty")
    rows = len(full_board)
    cols = len(full_board[0])
    if any(len(row) != cols for row in full_board):
        raise ValueError("board must be rectangular")
    flat = [int(v) for row in full_board for v in row]
    n = rows * cols
    if len(set(flat)) != n:
        raise ValueError("board values must be unique")
    if sorted(flat) != list(range(1, n + 1)):
        raise ValueError("board values must equal 1..N")


def mask_full_board(full_board: Board, masking_ratio: float, seed: int) -> Tuple[Board, List[Cell]]:
    validate_full_board(full_board)
    rows = len(full_board)
    cols = len(full_board[0])
    n = rows * cols
    mask_count = int(math.floor(n * masking_ratio))
    rng = random.Random(seed)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    masked_cells = cells[:mask_count]
    masked_set = set(masked_cells)
    masked = [[-1 if (r, c) in masked_set else full_board[r][c] for c in range(cols)] for r in range(rows)]
    return masked, masked_cells


def _iter_boards_from_json(path: Path) -> Iterable[FullBoardRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("boards", [])
    if isinstance(rows, dict):
        rows = [rows]
    for idx, item in enumerate(rows):
        board = item
        board_id = f"{path.stem}:{idx}"
        if isinstance(item, dict):
            board = item.get("board") or item.get("grid")
            board_id = str(item.get("board_id", board_id))
        if not isinstance(board, list):
            continue
        yield FullBoardRecord(board_id=board_id, board=board, source=str(path))


def _parse_grid_lines(lines: List[str]) -> Board:
    grid: Board = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            parts = [x.strip() for x in line.split(",")]
        else:
            parts = line.split()
        if not parts:
            continue
        grid.append([int(x) for x in parts])
    return grid


def _iter_boards_from_csv(path: Path) -> Iterable[FullBoardRecord]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and ("board" in reader.fieldnames or "grid" in reader.fieldnames):
            for idx, row in enumerate(reader):
                board_id = str(row.get("board_id", f"{path.stem}:{idx}"))
                raw = row.get("board") or row.get("grid") or ""
                try:
                    board = json.loads(raw)
                except Exception:
                    continue
                yield FullBoardRecord(board_id=board_id, board=board, source=str(path))
            return
    lines = path.read_text(encoding="utf-8").splitlines()
    grid = _parse_grid_lines(lines)
    if grid:
        yield FullBoardRecord(board_id=f"{path.stem}:0", board=grid, source=str(path))


def _iter_boards_from_txt(path: Path) -> Iterable[FullBoardRecord]:
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for idx, block in enumerate(blocks):
        lines = [x for x in block.splitlines() if x.strip()]
        board_id = f"{path.stem}:{idx}"
        if lines and lines[0].startswith("board_id:"):
            board_id = lines[0].split(":", 1)[1].strip() or board_id
            lines = lines[1:]
        try:
            grid = _parse_grid_lines(lines)
        except Exception:
            continue
        if grid:
            yield FullBoardRecord(board_id=board_id, board=grid, source=str(path))


def discover_full_boards(input_dir: Path) -> List[FullBoardRecord]:
    return discover_full_boards_with_audit(input_dir).boards


def discover_full_boards_with_audit(input_dir: Path) -> DiscoveryArtifacts:
    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")
    out: List[FullBoardRecord] = []
    invalid_reasons: Dict[str, int] = {}
    iterators = {
        ".json": _iter_boards_from_json,
        ".csv": _iter_boards_from_csv,
        ".txt": _iter_boards_from_txt,
    }
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in iterators:
            continue
        try:
            parsed_records = list(iterators[path.suffix.lower()](path))
        except Exception as exc:
            key = f"{path.suffix.lower()}_parse_error:{exc}"
            invalid_reasons[key] = invalid_reasons.get(key, 0) + 1
            continue
        for rec in parsed_records:
            try:
                validate_full_board(rec.board)
            except Exception as exc:
                key = str(exc)
                invalid_reasons[key] = invalid_reasons.get(key, 0) + 1
                continue
            out.append(rec)
    if not out:
        raise ValueError(f"no valid full-board data found under {input_dir}; reasons={invalid_reasons}")
    return DiscoveryArtifacts(boards=out, invalid_reasons=invalid_reasons)


def _rank_of_cell(candidates: Sequence[Dict[str, Any]], true_cell: Cell) -> Tuple[int, float, float]:
    for idx, cand in enumerate(candidates, start=1):
        if (cand["row"] - 1, cand["col"] - 1) == true_cell:
            return idx, float(cand["score"]), float(cand["confidence_1_to_100"])
    raise ValueError("true cell not found in candidates")


def aggregate_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {
            "top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top5_hit_rate": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "num_samples": 0,
        }
    return {
        "top1_hit_rate": sum(1 for x in ranks if x <= 1) / len(ranks),
        "top3_hit_rate": sum(1 for x in ranks if x <= 3) / len(ranks),
        "top5_hit_rate": sum(1 for x in ranks if x <= 5) / len(ranks),
        "mrr": mean(1.0 / x for x in ranks),
        "mean_rank": mean(ranks),
        "median_rank": float(median(ranks)),
        "num_samples": len(ranks),
    }


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {k: 1.0 / len(clipped) for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def random_weight_candidates(modules: List[str], trials: int, seed: int) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(trials):
        proposal = {m: rng.random() for m in modules}
        out.append(normalize_weights(proposal))
    return out


def run_weighted_eval(
    boards: Sequence[FullBoardRecord],
    weights: Dict[str, float],
    masking_ratio: float,
    repeats: int,
    seed: int,
    apply_reranker_stage: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    per_case: List[Dict[str, Any]] = []
    ranks: List[int] = []
    for board_idx, rec in enumerate(boards):
        for rep in range(repeats):
            masked, masked_cells = mask_full_board(rec.board, masking_ratio, seed + board_idx * 997 + rep)
            for target_cell in masked_cells:
                target_number = int(rec.board[target_cell[0]][target_cell[1]])
                result = _run_inference_detailed(
                    masked,
                    target_number=target_number,
                    source="mainline_eval",
                    module_weights=weights,
                    apply_reranker_stage=apply_reranker_stage,
                )
                rank, score, confidence = _rank_of_cell(result["candidate_cells"], target_cell)
                ranks.append(rank)
                per_case.append(
                    {
                        "board_id": rec.board_id,
                        "source": rec.source,
                        "repeat": rep,
                        "rows": len(rec.board),
                        "cols": len(rec.board[0]),
                        "target_number": target_number,
                        "true_row": target_cell[0] + 1,
                        "true_col": target_cell[1] + 1,
                        "rank": rank,
                        "score": score,
                        "confidence_1_to_100": confidence,
                        "top1_hit": int(rank <= 1),
                        "top3_hit": int(rank <= 3),
                        "top5_hit": int(rank <= 5),
                    }
                )
    return per_case, aggregate_metrics(ranks)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
