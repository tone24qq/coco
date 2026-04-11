from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.inference_service import compact_top10_response, run_inference, run_multi_target_inference


def infer_target_position(board: List[List[int]], target_number: int, source: str = "manual") -> Dict[str, Any]:
    return run_inference(board=board, target_number=target_number, source=source)


def infer_multi_target_positions(
    board: List[List[int]],
    target_numbers: List[int],
    source: str = "manual",
) -> Dict[str, Any]:
    result = run_multi_target_inference(
        board=board,
        target_numbers=target_numbers,
        source=source,
    )
    assignments = result.get("assignments", [])
    if not assignments:
        raise ValueError("multi-target inference returned no assignments")
    pseudo_candidates = [
        {
            "row": int(item["row"]),
            "col": int(item["col"]),
            "confidence_1_to_100": round(float(item.get("joint_score", 0.0)) * 100.0, 2),
        }
        for item in assignments
    ]
    pseudo_candidates.sort(key=lambda x: x["confidence_1_to_100"], reverse=True)
    return compact_top10_response({"candidate_cells": pseudo_candidates})


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
