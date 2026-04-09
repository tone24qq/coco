from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.inference_facade import (
        infer_target_position,
        map_score_to_confidence_1_100,
        validate_single_case_data,
    )

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

    inference_result = infer_target_position(masked_board, target_number, source="single_board_eval")
    candidate_cells = inference_result["candidate_cells"]

    all_scores = [float(c["score"]) for c in candidate_cells]
    min_score = min(all_scores)
    max_score = max(all_scores)

    top5 = []
    for rank, cell in enumerate(candidate_cells[:5], start=1):
        row_1b = int(cell["row"])
        col_1b = int(cell["col"])
        row_0b = row_1b - 1
        col_0b = col_1b - 1
        score = float(cell["score"])
        top5.append(
            {
                "rank": rank,
                "row_0_based": row_0b,
                "col_0_based": col_0b,
                "row_1_based": row_1b,
                "col_1_based": col_1b,
                "score": score,
                "confidence_1_to_100": map_score_to_confidence_1_100(score, min_score, max_score),
                "module_scores": cell.get("module_scores", {}),
            }
        )

    top1_cell = (top5[0]["row_0_based"], top5[0]["col_0_based"])
    top5_cells = {(c["row_0_based"], c["col_0_based"]) for c in top5}

    top1_hit = int(top1_cell == true_cell_0_based)
    top5_hit = int(true_cell_0_based in top5_cells)

    summary = {
        "target_number": target_number,
        "validation": validation,
        "true_cell_0_based": list(true_cell_0_based),
        "true_cell_1_based": [true_cell_0_based[0] + 1, true_cell_0_based[1] + 1],
        "top1_prediction": top5[0],
        "top5_predictions": top5,
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "top1_hit_rate_percent": top1_hit * 100,
        "top5_hit_rate_percent": top5_hit * 100,
        "notes": [
            "confidence_1_to_100 是依候選排序分數做單調映射，不是校準後機率",
            "映射規則: confidence = 1 + 99 * (score - min_score) / (max_score - min_score)",
        ],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
