from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_ranking_error_report(
    case_id: str,
    target_number: int,
    true_cell_1_based: Tuple[int, int],
    baseline_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gt_rank = None
    gt_cell = None
    for idx, c in enumerate(baseline_candidates, start=1):
        if (int(c["row"]), int(c["col"])) == true_cell_1_based:
            gt_rank = idx
            gt_cell = c
            break

    top5 = baseline_candidates[:5]
    pulling_down_modules: Dict[str, float] = {}
    if gt_cell and gt_cell.get("module_scores"):
        for name, score in gt_cell["module_scores"].items():
            top1_score = float(top5[0]["module_scores"].get(name, 0.0)) if top5 else 0.0
            pulling_down_modules[name] = round(top1_score - float(score), 6)

    top1_over_gt_score_gap = None
    if gt_cell and top5:
        top1_over_gt_score_gap = round(float(top5[0]["score"]) - float(gt_cell["score"]), 6)

    return {
        "case_id": case_id,
        "target_number": target_number,
        "gt_rank_baseline": gt_rank,
        "top5_competitors": top5,
        "modules_pulling_down_true_cell": pulling_down_modules,
        "score_gap_top1_minus_gt": top1_over_gt_score_gap,
        "rank_gap_top1_minus_gt": (gt_rank - 1) if gt_rank else None,
    }
