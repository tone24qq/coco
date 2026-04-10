from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.board_geometry import anti_diagonal_cells, cell_on_anti_diagonal, cell_on_main_diagonal, main_diagonal_cells

FEATURE_SCHEMA_VERSION = "ranking_features_v2"
TOP_KS = (1, 3, 5)


def _dense_ranks(values: List[float]) -> List[int]:
    sorted_unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
    return [rank_map[v] for v in values]


def _dense_rank_by_indices(all_scores: List[float], indices: List[int]) -> Dict[int, int]:
    if not indices:
        return {}
    sub_scores = [all_scores[i] for i in indices]
    sub_ranks = _dense_ranks(sub_scores)
    return {idx: rank for idx, rank in zip(indices, sub_ranks)}


def build_candidate_feature_rows(
    case_id: str,
    board_shape: Tuple[int, int],
    candidates: List[Dict[str, Any]],
    true_cell_1_based: Optional[Tuple[int, int]] = None,
    board: Optional[List[List[int]]] = None,
    target_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    _ = target_number
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
    row_to_indices: Dict[int, List[int]] = {}
    col_to_indices: Dict[int, List[int]] = {}
    main_diag_indices: List[int] = []
    anti_diag_indices: List[int] = []
    for idx, candidate in enumerate(candidates):
        row = int(candidate["row"])
        col = int(candidate["col"])
        row_to_indices.setdefault(row, []).append(idx)
        col_to_indices.setdefault(col, []).append(idx)
        zero_based = (row - 1, col - 1)
        if cell_on_main_diagonal(zero_based, rows, cols):
            main_diag_indices.append(idx)
        if cell_on_anti_diagonal(zero_based, rows, cols):
            anti_diag_indices.append(idx)

    row_relative_ranks = {
        row: _dense_rank_by_indices(baseline_scores, indices) for row, indices in row_to_indices.items()
    }
    col_relative_ranks = {
        col: _dense_rank_by_indices(baseline_scores, indices) for col, indices in col_to_indices.items()
    }
    main_diag_relative_ranks = _dense_rank_by_indices(baseline_scores, main_diag_indices)
    anti_diag_relative_ranks = _dense_rank_by_indices(baseline_scores, anti_diag_indices)

    for idx, candidate in enumerate(candidates):
        row = int(candidate["row"])
        col = int(candidate["col"])
        module_scores = candidate.get("module_scores", {})
        module_details = candidate.get("module_details", {})
        consensus = {
            f"module_consensus_top{k}": sum(1 for mod_set in module_topk[k] if idx in mod_set) for k in TOP_KS
        }
        directional = module_details.get("directional_consistency", {})
        line = module_details.get("line_consistency", {})
        global_details = module_details.get("global_assignment_prior", {})

        row_known_density = 0.0
        col_known_density = 0.0
        main_diag_known_density = 0.0
        anti_diag_known_density = 0.0
        if board is not None and board and board[0]:
            row_known_density = sum(1 for v in board[row - 1] if v != -1) / cols
            col_known_density = sum(1 for rr in range(rows) if board[rr][col - 1] != -1) / rows
            zero_based = (row - 1, col - 1)
            if cell_on_main_diagonal(zero_based, rows, cols):
                diag_cells = main_diagonal_cells(rows, cols)
                limit = max(len(diag_cells), 1)
                main_diag_known_density = sum(1 for rr, cc in diag_cells if board[rr][cc] != -1) / limit
            else:
                main_diag_known_density = 0.0
            if cell_on_anti_diagonal(zero_based, rows, cols):
                diag_cells = anti_diagonal_cells(rows, cols)
                limit = max(len(diag_cells), 1)
                anti_diag_known_density = sum(1 for rr, cc in diag_cells if board[rr][cc] != -1) / limit
            else:
                anti_diag_known_density = 0.0

        row_rel = row_relative_ranks.get(row, {}).get(idx, 1)
        col_rel = col_relative_ranks.get(col, {}).get(idx, 1)
        zero_based = (row - 1, col - 1)
        if cell_on_main_diagonal(zero_based, rows, cols):
            diag_rel = main_diag_relative_ranks.get(idx, 1)
        elif cell_on_anti_diagonal(zero_based, rows, cols):
            diag_rel = anti_diag_relative_ranks.get(idx, 1)
        else:
            diag_rel = 1

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
            **consensus,
            "directional_score": float(module_scores.get("directional_consistency", 0.0)),
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
            "line_score": float(module_scores.get("line_consistency", 0.0)),
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
                module_scores.get("global_assignment_prior", global_details.get("global_assignment_score", 0.5))
            ),
            "same_row_known_density": row_known_density,
            "same_col_known_density": col_known_density,
            "same_main_diag_known_density": main_diag_known_density,
            "same_anti_diag_known_density": anti_diag_known_density,
            "relative_rank_within_row": row_rel,
            "relative_rank_within_col": col_rel,
            "relative_rank_within_diag": diag_rel,
            "label": int(true_cell_1_based == (row, col)) if true_cell_1_based else None,
        }

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
