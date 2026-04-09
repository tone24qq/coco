from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


FEATURE_SCHEMA_VERSION = "ranking_features_v1"
TOP_KS = (1, 3, 5)


def _dense_ranks(values: List[float]) -> List[int]:
    sorted_unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
    return [rank_map[v] for v in values]


def build_candidate_feature_rows(
    case_id: str,
    board_shape: Tuple[int, int],
    candidates: List[Dict[str, Any]],
    true_cell_1_based: Optional[Tuple[int, int]] = None,
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
            **consensus,
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
