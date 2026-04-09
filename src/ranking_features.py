from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


FEATURE_SCHEMA_VERSION = "ranking_features_v1"
TOP_KS = (1, 3, 5)


def _dense_ranks(values: List[float]) -> List[int]:
    sorted_unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
    return [rank_map[v] for v in values]


def _known_density(board: List[List[int]], coords: List[Tuple[int, int]]) -> float:
    if not coords:
        return 0.0
    known = sum(1 for r, c in coords if board[r][c] != -1)
    return known / len(coords)


def _relative_rank(value: int, known_values: List[int]) -> float:
    if not known_values:
        return 0.5
    lower = sum(1 for v in known_values if v <= value)
    return lower / len(known_values)


def build_candidate_feature_rows(
    case_id: str,
    board_shape: Tuple[int, int],
    candidates: List[Dict[str, Any]],
    true_cell_1_based: Optional[Tuple[int, int]] = None,
    board: Optional[List[List[int]]] = None,
    target_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows, cols = board_shape
    if not candidates:
        return []

    baseline_scores = [float(c["score"]) for c in candidates]
    baseline_ranks = _dense_ranks(baseline_scores)
    top1_score = max(baseline_scores)
    top3_score = sorted(baseline_scores, reverse=True)[min(2, len(baseline_scores) - 1)]

    module_names = sorted(candidates[0].get("module_scores", {}).keys())
    module_score_maps = {m: [float(c["module_scores"].get(m, 0.0)) for c in candidates] for m in module_names}
    module_ranks = {m: _dense_ranks(scores) for m, scores in module_score_maps.items()}

    module_topk: Dict[int, List[set[int]]] = {
        k: [set(i for i, r in enumerate(module_ranks[m]) if r <= k) for m in module_names] for k in TOP_KS
    }

    center_r = (rows + 1) / 2.0
    center_c = (cols + 1) / 2.0
    max_center_dist = max(center_r + center_c, 1.0)

    out: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        row = int(candidate["row"])
        col = int(candidate["col"])
        module_scores = candidate.get("module_scores", {})
        module_details = candidate.get("module_details", {})
        directional = module_details.get("directional_consistency", {})
        line = module_details.get("line_consistency", {})
        global_detail = module_details.get("global_assignment_prior", {})

        consensus = {
            f"module_consensus_top{k}": sum(1 for mod_set in module_topk[k] if idx in mod_set) for k in TOP_KS
        }

        feature = {
            "case_id": case_id,
            "group_id": case_id,
            "row": row,
            "col": col,
            "baseline_score": baseline_scores[idx],
            "baseline_rank": baseline_ranks[idx],
            "score_gap_to_top1": top1_score - baseline_scores[idx],
            "score_gap_to_top3": top3_score - baseline_scores[idx],
            "candidate_count": len(candidates),
            "row_norm": row / rows,
            "col_norm": col / cols,
            "dist_to_center": (abs(row - center_r) + abs(col - center_c)) / max_center_dist,
            "is_border": int(row in (1, rows) or col in (1, cols)),
            "is_corner": int((row in (1, rows)) and (col in (1, cols))),
            "directional_score": float(
                module_scores.get("directional_consistency", directional.get("directional_score", 0.0))
            ),
            "left_order_score": float(directional.get("left_order_score", 0.5)),
            "right_order_score": float(directional.get("right_order_score", 0.5)),
            "up_order_score": float(directional.get("up_order_score", 0.5)),
            "down_order_score": float(directional.get("down_order_score", 0.5)),
            "left_distance_score": float(directional.get("left_distance_score", 0.5)),
            "right_distance_score": float(directional.get("right_distance_score", 0.5)),
            "up_distance_score": float(directional.get("up_distance_score", 0.5)),
            "down_distance_score": float(directional.get("down_distance_score", 0.5)),
            "row_balance_score": float(directional.get("row_balance_score", 0.5)),
            "col_balance_score": float(directional.get("col_balance_score", 0.5)),
            "line_score": float(module_scores.get("line_consistency", line.get("line_score", 0.0))),
            "row_residual_score": float(line.get("row_residual_score", 0.5)),
            "col_residual_score": float(line.get("col_residual_score", 0.5)),
            "main_diag_score": float(line.get("main_diag_score", 0.5)),
            "anti_diag_score": float(line.get("anti_diag_score", 0.5)),
            "row_monotonicity_score": float(line.get("row_monotonicity_score", 0.5)),
            "col_monotonicity_score": float(line.get("col_monotonicity_score", 0.5)),
            "diag_monotonicity_score": float(line.get("diag_monotonicity_score", 0.5)),
            "row_percentile_fit": float(line.get("row_percentile_fit", 0.5)),
            "col_percentile_fit": float(line.get("col_percentile_fit", 0.5)),
            "diag_percentile_fit": float(line.get("diag_percentile_fit", 0.5)),
            "global_assignment_score": float(
                module_scores.get("global_assignment_prior", global_detail.get("global_assignment_score", 0.5))
            ),
            **consensus,
            "label": int(true_cell_1_based == (row, col)) if true_cell_1_based else None,
        }

        if board is not None:
            r0, c0 = row - 1, col - 1
            row_coords = [(r0, x) for x in range(cols)]
            col_coords = [(x, c0) for x in range(rows)]
            main_coords = [(i, i) for i in range(min(rows, cols))] if r0 == c0 else []
            anti_coords = [(i, cols - 1 - i) for i in range(min(rows, cols))] if r0 + c0 == cols - 1 else []
            feature["same_row_known_density"] = _known_density(board, row_coords)
            feature["same_col_known_density"] = _known_density(board, col_coords)
            feature["same_main_diag_known_density"] = _known_density(board, main_coords) if main_coords else 0.0
            feature["same_anti_diag_known_density"] = _known_density(board, anti_coords) if anti_coords else 0.0
            if target_number is not None:
                row_known = [board[r0][x] for x in range(cols) if board[r0][x] != -1]
                col_known = [board[x][c0] for x in range(rows) if board[x][c0] != -1]
                diag_known = [board[a][b] for a, b in (main_coords or anti_coords) if board[a][b] != -1]
                feature["relative_rank_within_row"] = _relative_rank(target_number, row_known)
                feature["relative_rank_within_col"] = _relative_rank(target_number, col_known)
                feature["relative_rank_within_diag"] = _relative_rank(target_number, diag_known)
            else:
                feature["relative_rank_within_row"] = 0.5
                feature["relative_rank_within_col"] = 0.5
                feature["relative_rank_within_diag"] = 0.5
        else:
            feature["same_row_known_density"] = 0.0
            feature["same_col_known_density"] = 0.0
            feature["same_main_diag_known_density"] = 0.0
            feature["same_anti_diag_known_density"] = 0.0
            feature["relative_rank_within_row"] = 0.5
            feature["relative_rank_within_col"] = 0.5
            feature["relative_rank_within_diag"] = 0.5

        for m in module_names:
            feature[f"module_score_{m}"] = float(module_scores.get(m, 0.0))
            feature[f"module_rank_{m}"] = module_ranks[m][idx]

        out.append(feature)

    return out


def feature_columns_from_rows(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    exclude = {"case_id", "group_id", "row", "col", "label"}
    return [k for k in rows[0].keys() if k not in exclude]
