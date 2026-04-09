from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.inference_config import load_module_weights
from src.scoring_modules import MODULES, Cell, ModuleScoreResult


@dataclass
class ParsedBoard:
    rows: int
    cols: int
    opened_numbers: Dict[int, Cell]
    unopened_cells: List[Cell]


class InferenceError(ValueError):
    """Domain validation error for inference."""


def parse_board_input(board: List[List[int]]) -> ParsedBoard:
    if not board or not board[0]:
        raise InferenceError("board must be non-empty")
    rows = len(board)
    cols = len(board[0])
    opened_numbers: Dict[int, Cell] = {}
    unopened_cells: List[Cell] = []

    for r, row in enumerate(board):
        if len(row) != cols:
            raise InferenceError("board must be rectangular")
        for c, value in enumerate(row):
            if value == -1:
                unopened_cells.append((r, c))
                continue
            if value <= 0:
                raise InferenceError("opened cells must be positive integers")
            if value in opened_numbers:
                raise InferenceError(f"duplicate opened number detected: {value}")
            opened_numbers[value] = (r, c)

    return ParsedBoard(rows, cols, opened_numbers, unopened_cells)


def compute_remaining_numbers(parsed: ParsedBoard) -> List[int]:
    n_total = parsed.rows * parsed.cols
    allowed = set(range(1, n_total + 1))
    opened = set(parsed.opened_numbers.keys())
    invalid = sorted(opened - allowed)
    if invalid:
        raise InferenceError(f"opened number out of range 1..N: {invalid[0]}")
    return sorted(allowed - opened)


def validate_target_number(
    target_number: int,
    parsed: ParsedBoard,
    remaining_numbers: List[int],
) -> Tuple[str, Optional[Cell]]:
    n_total = parsed.rows * parsed.cols
    if not (1 <= target_number <= n_total):
        raise InferenceError("target_number out of range 1..N")

    if target_number in parsed.opened_numbers:
        return "already_opened", parsed.opened_numbers[target_number]

    if target_number not in remaining_numbers:
        raise InferenceError("target_number must be in remaining numbers when not opened")

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
            raise InferenceError(f"Unknown module in weights: {module_name}")
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


def map_score_to_confidence_1_100(score: float, min_score: float, max_score: float) -> float:
    if max_score - min_score < 1e-12:
        return 50.0
    scaled = (score - min_score) / (max_score - min_score)
    return round(1.0 + 99.0 * scaled, 2)


def build_explanation(
    rows: int,
    cols: int,
    target_number: int,
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


def run_inference(
    board: List[List[int]],
    target_number: int,
    source: str,
    module_weights: Optional[Dict[str, float]] = None,
    version: str = "v1",
) -> Dict[str, Any]:
    parsed = parse_board_input(board)
    remaining = compute_remaining_numbers(parsed)
    status, opened_cell = validate_target_number(target_number, parsed, remaining)

    unopened_cells_payload = [{"row": r + 1, "col": c + 1} for r, c in parsed.unopened_cells]

    if status == "already_opened" and opened_cell is not None:
        return {
            "status": "already_opened",
            "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
            "target_number": target_number,
            "remaining_numbers": remaining,
            "unopened_cells": unopened_cells_payload,
            "best_cell": {
                "row": opened_cell[0] + 1,
                "col": opened_cell[1] + 1,
                "score": 1.0,
                "confidence_1_to_100": 100.0,
            },
            "candidate_cells": [],
            "confidence_score": 1.0,
            "reasoning": [
                f"盤面總格數為 {parsed.rows * parsed.cols}，合法數字集合為 1..{parsed.rows * parsed.cols}",
                f"target_number={target_number} 已經在已開格",
            ],
            "module_contributions": {},
            "metadata": {
                "score_type": "position_confidence_score",
                "confidence_type": "deterministic_when_already_opened",
                "confidence_1_to_100_type": "fixed_100_for_already_opened",
                "confidence_1_to_100_is_probability": False,
                "source": source,
                "version": version,
            },
        }

    if not parsed.unopened_cells:
        raise InferenceError("board has no unopened cells")

    candidates = build_cell_candidates(parsed.unopened_cells)
    scored, weights, module_explanations = score_candidates(board, candidates, target_number, module_weights)
    ranked = rank_candidates(scored)
    best = ranked[0]

    all_scores = [float(c["score"]) for c in ranked]
    min_score = min(all_scores)
    max_score = max(all_scores)

    candidate_cells = []
    for cell in ranked:
        score = round(float(cell["score"]), 6)
        candidate_cells.append(
            {
                "row": cell["cell"][0] + 1,
                "col": cell["cell"][1] + 1,
                "score": score,
                "confidence_1_to_100": map_score_to_confidence_1_100(score, min_score, max_score),
                "module_scores": {
                    k: round(float(v), 6) for k, v in sorted(cell["module_scores"].items())
                },
            }
        )

    reasoning = build_explanation(
        parsed.rows,
        parsed.cols,
        target_number,
        len(parsed.unopened_cells),
        weights,
        best["cell"],
        module_explanations,
    )

    best_score = round(float(best["score"]), 6)
    return {
        "status": "ok",
        "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
        "target_number": target_number,
        "remaining_numbers": remaining,
        "unopened_cells": unopened_cells_payload,
        "best_cell": {
            "row": best["cell"][0] + 1,
            "col": best["cell"][1] + 1,
            "score": best_score,
            "confidence_1_to_100": map_score_to_confidence_1_100(best_score, min_score, max_score),
        },
        "candidate_cells": candidate_cells,
        "confidence_score": best_score,
        "reasoning": reasoning,
        "module_contributions": weights,
        "metadata": {
            "score_type": "position_confidence_score",
            "confidence_type": "monotonic_rank_score_mapping",
            "confidence_1_to_100_type": "monotonic_mapped_score_non_calibrated",
            "confidence_1_to_100_is_probability": False,
            "source": source,
            "version": version,
        },
    }
