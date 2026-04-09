from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.inference_service import run_inference


def infer_target_position(board: List[List[int]], target_number: int, source: str = "manual") -> Dict[str, Any]:
    return run_inference(board=board, target_number=target_number, source=source)


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
    masking_ratio: float = 0.5,
) -> Dict[str, object]:
    if not full_board or not full_board[0]:
        raise ValueError("full_board must be non-empty")
    rows = len(full_board)
    cols = len(full_board[0])
    if any(len(row) != cols for row in full_board):
        raise ValueError("full_board must be rectangular")
    flat_full = [v for row in full_board for v in row]
    n_total = rows * cols
    if sorted(flat_full) != list(range(1, n_total + 1)):
        raise ValueError("full_board must contain 1..N exactly once")

    if len(masked_board) != rows or any(len(row) != cols for row in masked_board):
        raise ValueError("masked_board shape must equal full_board shape")

    masked_count = sum(1 for row in masked_board for v in row if v == -1)
    expected_masked = int(n_total * masking_ratio)
    if masked_count != expected_masked:
        raise ValueError(f"masked_board must mask exactly {expected_masked} cells for ratio={masking_ratio}")

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
    tr, tc = true_cell_0_based
    if masked_board[tr][tc] != -1:
        raise ValueError("true_cell_0_based must be masked in masked_board")

    return {
        "shape": [rows, cols],
        "masked_count": masked_count,
        "target_cell_0_based": list(target_pos),
        "target_cell_1_based": [target_pos[0] + 1, target_pos[1] + 1],
    }
