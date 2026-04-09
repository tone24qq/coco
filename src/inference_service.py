from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.inference_config import load_module_weights
from src.scoring_modules import MODULES, Cell, ModuleScoreResult


@dataclass
class ParsedBoard:
    rows: int
    cols: int
    opened_numbers: Dict[int, Cell]
    unopened_cells: List[Cell]


def parse_board_input(board: List[List[int]]) -> ParsedBoard:
    if not board or not board[0]:
        raise ValueError("board must be non-empty")
    rows = len(board)
    cols = len(board[0])
    opened_numbers: Dict[int, Cell] = {}
    unopened_cells: List[Cell] = []

    for r, row in enumerate(board):
        if len(row) != cols:
            raise ValueError("board must be rectangular")
        for c, value in enumerate(row):
            if value == -1:
                unopened_cells.append((r, c))
                continue
            if value <= 0:
                raise ValueError("opened cells must be positive integers")
            if value in opened_numbers:
                raise ValueError(f"duplicate opened number detected: {value}")
            opened_numbers[value] = (r, c)

    return ParsedBoard(rows, cols, opened_numbers, unopened_cells)


def compute_remaining_numbers(parsed: ParsedBoard) -> List[int]:
    n_total = parsed.rows * parsed.cols
    allowed = set(range(1, n_total + 1))
    opened = set(parsed.opened_numbers.keys())
    invalid = sorted(opened - allowed)
    if invalid:
        raise ValueError(f"opened number out of range 1..N: {invalid[0]}")
    return sorted(allowed - opened)


def validate_target_number(
    target_number: int,
    parsed: ParsedBoard,
    remaining_numbers: List[int],
) -> Tuple[str, Optional[Cell]]:
    n_total = parsed.rows * parsed.cols
    if not (1 <= target_number <= n_total):
        raise ValueError("target_number out of range 1..N")

    if target_number in parsed.opened_numbers:
        return "already_opened", parsed.opened_numbers[target_number]

    if target_number not in remaining_numbers:
        raise ValueError("target_number must be in remaining numbers when not opened")

    return "ok", None


def build_cell_candidates(
    unopened_cells: List[Cell],
) -> List[Dict[str, object]]:
    return [
        {
            "cell": cell,
            "score": 0.0,
            "module_scores": {},
        }
        for cell in unopened_cells
    ]


def _normalize_scores(raw_scores: Dict[Cell, float]) -> Dict[Cell, float]:
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-12:
        return {k: 1.0 for k in raw_scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in raw_scores.items()}


def score_candidates(
    board: List[List[int]],
    candidates: List[Dict[str, object]],
    target_number: int,
    module_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float], List[str]]:
    weights = module_weights or load_module_weights()
    explanations: List[str] = []

    for module_name, weight in weights.items():
        if module_name not in MODULES:
            raise ValueError(f"Unknown module in weights: {module_name}")
        result: ModuleScoreResult = MODULES[module_name].score(
            board,
            [c["cell"] for c in candidates],
            target_number,
        )
        explanations.append(result.explanation)
        normalized = _normalize_scores(result.scores)
        for c in candidates:
            cell = c["cell"]
            module_score = float(normalized.get(cell, 0.0))
            c["module_scores"][module_name] = module_score
            c["score"] += module_score * weight

    return candidates, weights, explanations


def rank_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def build_explanation(
    rows: int,
    cols: int,
    target_number: int,
    remaining_numbers: List[int],
    unopened_count: int,
    module_weights: Dict[str, float],
    best_cell: Optional[Tuple[int, int]],
    module_explanations: List[str],
) -> List[str]:
    reasoning = [
        f"盤面總格數為 {rows * cols}，因此合法數字集合為 1..{rows * cols}",
        f"target_number={target_number} 尚未出現在已開格",
        f"目前共有 {unopened_count} 個未開格",
        "模組加權為 " + ", ".join(f"{k}={v:.2f}" for k, v in module_weights.items()),
    ]
    reasoning.extend(module_explanations)
    if best_cell is not None:
        reasoning.append(f"綜合模組分數後，row={best_cell[0] + 1}, col={best_cell[1] + 1} 最高")
    return reasoning
