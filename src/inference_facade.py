from __future__ import annotations

from typing import Dict, List, Tuple

from src.inference_service import (
    build_cell_candidates,
    build_explanation,
    compute_remaining_numbers,
    parse_board_input,
    rank_candidates,
    score_candidates,
    validate_target_number,
)


def infer_target_position(board: List[List[int]], target_number: int, source: str = "manual") -> Dict[str, object]:
    parsed = parse_board_input(board)
    remaining = compute_remaining_numbers(parsed)
    status, opened_cell = validate_target_number(target_number, parsed, remaining)

    unopened_cells_payload = [
        {"row": r + 1, "col": c + 1} for r, c in parsed.unopened_cells
    ]

    if status == "already_opened" and opened_cell is not None:
        return {
            "status": "already_opened",
            "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
            "target_number": target_number,
            "remaining_numbers": remaining,
            "unopened_cells": unopened_cells_payload,
            "best_cell": {"row": opened_cell[0] + 1, "col": opened_cell[1] + 1, "score": 1.0},
            "candidate_cells": [],
            "confidence_score": 1.0,
            "reasoning": [
                f"盤面總格數為 {parsed.rows * parsed.cols}，合法數字集合為 1..{parsed.rows * parsed.cols}",
                f"target_number={target_number} 已經在已開格",
            ],
            "module_contributions": {},
            "metadata": {
                "score_type": "position_confidence",
                "source": source,
                "version": "v1",
            },
        }

    if not parsed.unopened_cells:
        raise ValueError("board has no unopened cells")

    candidates = build_cell_candidates(parsed.unopened_cells)
    scored, weights, module_explanations = score_candidates(
        board,
        candidates,
        target_number,
    )
    ranked = rank_candidates(scored)
    best = ranked[0]

    reasoning = build_explanation(
        parsed.rows,
        parsed.cols,
        target_number,
        remaining,
        len(parsed.unopened_cells),
        weights,
        best["cell"],
        module_explanations,
    )

    candidate_cells = [
        {
            "row": cell["cell"][0] + 1,
            "col": cell["cell"][1] + 1,
            "score": round(float(cell["score"]), 6),
            "module_scores": {
                k: round(float(v), 6) for k, v in sorted(cell["module_scores"].items())
            },
        }
        for cell in ranked
    ]

    return {
        "status": "ok",
        "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
        "target_number": target_number,
        "remaining_numbers": remaining,
        "unopened_cells": unopened_cells_payload,
        "best_cell": {
            "row": best["cell"][0] + 1,
            "col": best["cell"][1] + 1,
            "score": round(float(best["score"]), 6),
        },
        "candidate_cells": candidate_cells,
        "confidence_score": round(float(best["score"]), 6),
        "reasoning": reasoning,
        "module_contributions": weights,
        "metadata": {
            "score_type": "position_confidence",
            "source": source,
            "version": "v1",
        },
    }


def map_score_to_confidence_1_100(score: float, min_score: float, max_score: float) -> float:
    if max_score - min_score < 1e-12:
        return 50.0
    scaled = (score - min_score) / (max_score - min_score)
    return round(1.0 + 99.0 * scaled, 2)


def validate_single_case_data(
    full_board: List[List[int]],
    masked_board: List[List[int]],
    target_number: int,
    true_cell_0_based: Tuple[int, int],
) -> Dict[str, object]:
    if len(full_board) != 8 or any(len(row) != 10 for row in full_board):
        raise ValueError("full_board must be 8x10")
    flat_full = [v for row in full_board for v in row]
    if sorted(flat_full) != list(range(1, 81)):
        raise ValueError("full_board must contain 1..80 exactly once")

    if len(masked_board) != 8 or any(len(row) != 10 for row in masked_board):
        raise ValueError("masked_board must be 8x10")

    masked_count = sum(1 for row in masked_board for v in row if v == -1)
    if masked_count != 40:
        raise ValueError("masked_board must mask exactly 40 cells for 50% masking")

    target_pos = None
    for r, row in enumerate(full_board):
        for c, value in enumerate(row):
            if value == target_number:
                target_pos = (r, c)
                break
        if target_pos is not None:
            break

    if target_pos is None:
        raise ValueError("target_number is not present in full_board")

    if target_pos != true_cell_0_based:
        raise ValueError("true_cell_0_based does not match full_board location")

    return {
        "shape": [8, 10],
        "masked_count": masked_count,
        "target_cell_0_based": list(target_pos),
        "target_cell_1_based": [target_pos[0] + 1, target_pos[1] + 1],
    }
