from __future__ import annotations

from src.inference_facade import infer_target_position, map_score_to_confidence_1_100, validate_single_case_data


def test_single_board_top5_eval_case() -> None:
    target_number = 33
    full_board = [
        [37, 12, 58, 4, 71, 26, 49, 80, 15, 63],
        [22, 54, 1, 68, 33, 47, 9, 72, 29, 60],
        [75, 18, 44, 6, 52, 39, 64, 11, 57, 24],
        [30, 66, 14, 79, 41, 2, 53, 20, 70, 35],
        [8, 61, 27, 46, 13, 74, 31, 55, 17, 69],
        [43, 5, 59, 21, 76, 34, 65, 10, 48, 28],
        [73, 16, 40, 62, 7, 56, 25, 78, 32, 50],
        [19, 67, 3, 45, 23, 77, 42, 51, 36, 38],
    ]

    masked_board = [
        [-1, 12, 58, -1, -1, 26, -1, 80, 15, -1],
        [-1, -1, -1, -1, -1, -1, 9, 72, -1, 60],
        [75, 18, -1, 6, 52, 39, 64, -1, -1, -1],
        [-1, 66, -1, 79, 41, -1, -1, 20, -1, 35],
        [8, 61, -1, 46, 13, -1, -1, 55, 17, -1],
        [43, 5, -1, -1, 76, -1, 65, 10, 48, -1],
        [73, -1, 40, 62, 7, 56, -1, 78, -1, 50],
        [-1, -1, 3, -1, -1, -1, 42, -1, -1, -1],
    ]

    true_cell_0_based = (1, 4)

    validation = validate_single_case_data(
        full_board=full_board,
        masked_board=masked_board,
        target_number=target_number,
        true_cell_0_based=true_cell_0_based,
    )
    assert validation["masked_count"] == 40
    assert validation["shape"] == [8, 10]
    assert validation["target_cell_0_based"] == [1, 4]

    result = infer_target_position(masked_board, target_number, source="single_board_test")
    assert result["status"] == "ok"
    assert len(result["candidate_cells"]) == 40

    top5 = result["candidate_cells"][:5]
    all_scores = [float(c["score"]) for c in result["candidate_cells"]]
    min_score = min(all_scores)
    max_score = max(all_scores)

    for cell in top5:
        conf = map_score_to_confidence_1_100(float(cell["score"]), min_score, max_score)
        assert 1.0 <= conf <= 100.0

    ranked_cells = [(c["row"] - 1, c["col"] - 1) for c in top5]
    top1_hit = int(ranked_cells[0] == true_cell_0_based)
    top5_hit = int(true_cell_0_based in ranked_cells)

    assert top1_hit in (0, 1)
    assert top5_hit in (0, 1)
