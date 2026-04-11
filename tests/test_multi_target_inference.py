from __future__ import annotations

from unittest.mock import patch

from src.inference_service import run_multi_target_inference, solve_joint_assignment


def _fake_single_result(top1_row: int, top1_col: int, alt_row: int, alt_col: int) -> dict:
    return {
        "status": "ok",
        "candidate_cells": [
            {
                "row": top1_row,
                "col": top1_col,
                "score": 0.9,
                "contradiction_penalty": 0.0,
                "gate_multiplier": 1.0,
            },
            {
                "row": alt_row,
                "col": alt_col,
                "score": 0.85,
                "contradiction_penalty": 0.0,
                "gate_multiplier": 1.0,
            },
        ],
    }


def test_joint_assignment_dedups_duplicate_individual_top1() -> None:
    board = [[1, -1], [-1, 4]]
    with patch(
        "src.inference_service._run_inference_detailed",
        side_effect=[
            _fake_single_result(1, 2, 2, 1),
            _fake_single_result(1, 2, 2, 1),
        ],
    ):
        result = run_multi_target_inference(board, [2, 3], source="t")
    assigned_cells = {(a["row"], a["col"]) for a in result["assignments"]}
    assert len(assigned_cells) == 2
    assert result["metadata"]["duplicate_top1_count_before_assignment"] == 1
    assert result["metadata"]["duplicate_top1_count_after_assignment"] == 0


def test_joint_assignment_keeps_strong_individual_top1() -> None:
    board = [[1, -1], [-1, 4]]
    with patch(
        "src.inference_service._run_inference_detailed",
        side_effect=[
            {
                "status": "ok",
                "candidate_cells": [
                    {"row": 1, "col": 2, "score": 0.99, "contradiction_penalty": 0.0, "gate_multiplier": 1.0},
                    {"row": 2, "col": 1, "score": 0.2, "contradiction_penalty": 0.0, "gate_multiplier": 1.0},
                ],
            },
            {
                "status": "ok",
                "candidate_cells": [
                    {"row": 1, "col": 2, "score": 0.8, "contradiction_penalty": 0.0, "gate_multiplier": 1.0},
                    {"row": 2, "col": 1, "score": 0.79, "contradiction_penalty": 0.0, "gate_multiplier": 1.0},
                ],
            },
        ],
    ):
        result = run_multi_target_inference(board, [2, 3], source="t")
    t2 = next(a for a in result["assignments"] if a["target_number"] == 2)
    assert (t2["row"], t2["col"]) == (1, 2)
    assert t2["was_reassigned_from_individual_top1"] is False


def test_solve_joint_assignment_exact_path() -> None:
    assigned, mode = solve_joint_assignment(
        target_numbers=[2, 3],
        unopened_cells=[(0, 1), (1, 0)],
        cost_matrix=[[0.1, 0.9], [0.9, 0.1]],
        assignment_mode="exact",
    )
    assert mode == "exact"
    assert len(set(assigned.values())) == 2


def test_solve_joint_assignment_greedy_fallback_path() -> None:
    with patch("src.inference_service.linear_sum_assignment", None):
        assigned, mode = solve_joint_assignment(
            target_numbers=[2, 3],
            unopened_cells=[(0, 1), (1, 0)],
            cost_matrix=[[0.1, 0.9], [0.9, 0.1]],
            assignment_mode="exact",
        )
    assert mode == "greedy"
    assert len(set(assigned.values())) == 2


def test_multi_target_assignment_count_matches_targets() -> None:
    result = run_multi_target_inference([[1, -1], [-1, 4]], [2, 3], source="t")
    assert len(result["assignments"]) == 2
    assert len({(a["row"], a["col"]) for a in result["assignments"]}) == 2
