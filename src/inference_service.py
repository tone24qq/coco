from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

from src.inference_config import (
    load_aggregator_config,
    load_fast_path_config,
    load_joint_assignment_config,
    load_module_settings,
    load_module_weights,
)
from src.competitive_fusion import (
    aggregate_topk_votes,
    build_meta_judge_feature_row,
    borda_scores,
    compute_dense_ranks,
    compute_rank_entropy_like,
    compute_vote_signals,
    load_meta_judge_artifact,
    normalize_scores_per_module,
    rrf_scores,
    score_with_logistic_artifact,
    validate_meta_judge_artifact,
)
from src.ranking_features import build_candidate_feature_rows
from src.reranker import apply_reranker, load_reranker_artifact
from src.scoring_modules import (
    Cell,
    GlobalAssignmentPriorModule,
    ModuleScoreResult,
    PairwiseConditionalConsistencyModule,
    build_modules,
)
from src.scoring_modules import linear_sum_assignment
from src.vector_modules import support_profile, with_fairness_diagnostics


@dataclass
class ParsedBoard:
    rows: int
    cols: int
    opened_numbers: Dict[int, Cell]
    unopened_cells: List[Cell]


class InferenceError(ValueError):
    """Domain validation error for inference."""


def parse_board_input(board: List[List[int]]) -> ParsedBoard:
    if not board or not board[0]:
        raise InferenceError("board must be non-empty")
    rows = len(board)
    cols = len(board[0])
    opened_numbers: Dict[int, Cell] = {}
    unopened_cells: List[Cell] = []

    for r, row in enumerate(board):
        if len(row) != cols:
            raise InferenceError("board must be rectangular")
        for c, value in enumerate(row):
            if value == -1:
                unopened_cells.append((r, c))
                continue
            if value <= 0:
                raise InferenceError("opened cells must be positive integers")
            if value in opened_numbers:
                raise InferenceError(f"duplicate opened number detected: {value}")
            opened_numbers[value] = (r, c)

    return ParsedBoard(rows, cols, opened_numbers, unopened_cells)


def compute_remaining_numbers(parsed: ParsedBoard) -> List[int]:
    n_total = parsed.rows * parsed.cols
    allowed = set(range(1, n_total + 1))
    opened = set(parsed.opened_numbers.keys())
    invalid = sorted(opened - allowed)
    if invalid:
        raise InferenceError(f"opened number out of range 1..N: {invalid[0]}")
    return sorted(allowed - opened)


def validate_target_number(
    target_number: int,
    parsed: ParsedBoard,
    remaining_numbers: List[int],
) -> Tuple[str, Optional[Cell]]:
    n_total = parsed.rows * parsed.cols
    if not (1 <= target_number <= n_total):
        raise InferenceError("target_number out of range 1..N")

    if target_number in parsed.opened_numbers:
        return "already_opened", parsed.opened_numbers[target_number]

    if target_number not in remaining_numbers:
        raise InferenceError("target_number must be in remaining numbers when not opened")

    return "ok", None


def build_cell_candidates(
    unopened_cells: List[Cell],
) -> List[Dict[str, object]]:
    return [
        {
            "cell": cell,
            "score": 0.0,
            "module_scores": {},
            "module_details": {},
            "module_informative": {},
        }
        for cell in unopened_cells
    ]


def _normalize_scores(raw_scores: Dict[Cell, float], mode: str = "disabled") -> Dict[Cell, float]:
    if not raw_scores:
        return {}
    if mode == "disabled":
        return {k: float(v) for k, v in raw_scores.items()}
    if mode == "light":
        mean_v = sum(raw_scores.values()) / len(raw_scores)
        return {k: _clip(0.5 + (float(v) - mean_v)) for k, v in raw_scores.items()}
    if mode != "minmax":
        raise InferenceError(f"Unknown normalization mode: {mode}")
    values = list(raw_scores.values())
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-12:
        return {k: 0.5 for k in raw_scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in raw_scores.items()}


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _get_informative_value(result: ModuleScoreResult, cell: Cell) -> float:
    informative = getattr(result, "informative_cells", None)
    if not informative:
        return 1.0
    return _clip(float(informative.get(cell, 1.0)))


def _validate_committee_stage1_modules(weights: Dict[str, float]) -> None:
    stage1_modules = set(weights.keys())
    banned = {"global_assignment_prior", "pairwise_conditional_consistency", "prior_model"}
    invalid = sorted(stage1_modules & banned)
    if invalid:
        raise InferenceError(f"committee stage-1 cannot include modules: {invalid}")
    if "structural_consistency" in stage1_modules and (
        "directional_consistency" in stage1_modules or "line_consistency" in stage1_modules
    ):
        raise InferenceError("structural_consistency cannot be enabled with directional_consistency/line_consistency")
    if "directional_consistency" in stage1_modules or "line_consistency" in stage1_modules:
        raise InferenceError("committee stage-1 must use structural_consistency instead of directional/line modules")


def score_candidates_raw(
    board: List[List[int]],
    candidates: List[Dict[str, object]],
    target_number: int,
    module_weights: Optional[Dict[str, float]] = None,
    module_settings: Optional[Dict[str, Dict[str, object]]] = None,
    normalization_mode: str = "disabled",
) -> Tuple[List[Dict[str, object]], Dict[str, float], List[str]]:
    weights = module_weights or load_module_weights()
    settings = module_settings if module_settings is not None else load_module_settings()
    fast_cfg = load_fast_path_config()
    runtime_mode = str(fast_cfg.get("runtime_mode", "fast"))
    if bool(fast_cfg.get("enabled", True)):
        use_numba = bool(fast_cfg.get("use_numba", True))
        for module_name in ("logic_rule", "prior_model", "directional_consistency", "line_consistency"):
            settings.setdefault(module_name, {})
            settings[module_name].setdefault("fast_enabled", use_numba)
        settings.setdefault("global_assignment_prior", {})
        global_top_m = int(fast_cfg.get("global_assignment_top_m_candidates", 4))
        settings["global_assignment_prior"].setdefault("top_m_candidates", global_top_m)
        settings["global_assignment_prior"].setdefault(
            "exact_max_candidates",
            int(fast_cfg.get("exact_max_candidates", 20)),
        )
        if runtime_mode == "fast":
            settings["global_assignment_prior"].setdefault("assignment_mode", "greedy")
        settings.setdefault("pairwise_conditional_consistency", {})
        settings["pairwise_conditional_consistency"].setdefault("runtime_mode", runtime_mode)
        settings["pairwise_conditional_consistency"].setdefault(
            "candidate_top_n",
            int(fast_cfg.get("pairwise_candidate_top_n", 8)),
        )
        settings["pairwise_conditional_consistency"].setdefault("global_assignment_mode", "greedy")
        settings["pairwise_conditional_consistency"].setdefault("global_assignment_top_m_candidates", global_top_m)
        settings["pairwise_conditional_consistency"].setdefault(
            "pairwise_seed_top_n",
            int(fast_cfg.get("pairwise_seed_top_n", 8)),
        )
        settings["pairwise_conditional_consistency"].setdefault(
            "pairwise_seed_modules",
            list(
                fast_cfg.get(
                    "pairwise_seed_modules",
                    [
                        "logic_rule",
                        "directional_consistency",
                        "line_consistency",
                        "difference_trend",
                        "skip_patterns",
                    ],
                )
            ),
        )
    modules = build_modules(settings)
    explanations: List[str] = []
    pairwise_name = "pairwise_conditional_consistency"
    candidate_cells = [c["cell"] for c in candidates]
    for module_name, weight in weights.items():
        if module_name == pairwise_name:
            continue
        if module_name not in modules:
            raise InferenceError(f"Unknown module in weights: {module_name}")
        result: ModuleScoreResult = modules[module_name].score(board, candidate_cells, target_number)
        explanations.append(result.explanation)
        normalized = _normalize_scores(result.scores, mode=normalization_mode)
        for c in candidates:
            c.setdefault("module_scores", {})
            c.setdefault("module_details", {})
            c.setdefault("module_informative", {})
            cell = c["cell"]
            module_score = float(normalized.get(cell, 0.0))
            c["module_scores"][module_name] = module_score
            if result.details:
                c["module_details"][module_name] = result.details.get(cell, {})
            else:
                c["module_details"][module_name] = {}
            if not isinstance(c["module_details"][module_name], dict):
                c["module_details"][module_name] = {}
            if "zone_type" not in c["module_details"][module_name]:
                fair = with_fairness_diagnostics(
                    board,
                    cell,
                    raw_score=float(result.scores.get(cell, 0.5)),
                    bias_corrected_score=module_score,
                    local_radius=1,
                )
                c["module_details"][module_name].update(fair)
            c["module_informative"][module_name] = _get_informative_value(result, cell)
            c["score"] += module_score * weight

    if pairwise_name in weights:
        if pairwise_name not in modules:
            raise InferenceError(f"Unknown module in weights: {pairwise_name}")
        pairwise_seed_modules = list(
            settings.get(pairwise_name, {}).get(
                "pairwise_seed_modules",
                [
                    "logic_rule",
                    "directional_consistency",
                    "line_consistency",
                    "difference_trend",
                    "skip_patterns",
                ],
            )
        )
        if not pairwise_seed_modules:
            raise InferenceError("pairwise_seed_modules must be non-empty")
        if pairwise_seed_modules == ["__all_enabled_modules__"]:
            pairwise_seed_modules = [m for m in weights if m != pairwise_name]
        elif pairwise_seed_modules == ["__auto_top_competitors__"]:
            non_pairwise_modules = [m for m in weights if m != pairwise_name]
            neutral_cell_scores: Dict[Cell, float] = {}
            for c in candidates:
                cell = c["cell"]
                vals = [float(c["module_scores"].get(m, 0.0)) for m in non_pairwise_modules]
                neutral_cell_scores[cell] = sum(vals) / max(len(vals), 1)
            top_cells = {
                cell
                for cell, _ in sorted(neutral_cell_scores.items(), key=lambda x: x[1], reverse=True)[
                    : max(1, min(5, len(neutral_cell_scores)))
                ]
            }
            module_alignment: List[Tuple[str, float]] = []
            for m in non_pairwise_modules:
                vals = [float(c["module_scores"].get(m, 0.0)) for c in candidates if c["cell"] in top_cells]
                module_alignment.append((m, sum(vals) / max(len(vals), 1)))
            module_alignment.sort(key=lambda x: x[1], reverse=True)
            pairwise_seed_modules = [m for m, _ in module_alignment[: max(1, min(4, len(module_alignment)))]]
        known_modules = set(build_modules().keys())
        missing_seed_modules = [m for m in pairwise_seed_modules if m not in known_modules]
        if missing_seed_modules:
            raise InferenceError(f"pairwise_seed_modules contain unknown modules: {missing_seed_modules}")
        seed_weight_sum = sum(float(weights.get(m, 0.0)) for m in pairwise_seed_modules)
        seed_scores: Dict[Cell, float] = {}
        for c in candidates:
            cell = c["cell"]
            v = 0.0
            for m in pairwise_seed_modules:
                v += float(c["module_scores"].get(m, 0.0)) * float(weights.get(m, 0.0))
            seed_scores[cell] = v / max(seed_weight_sum, 1e-12)
        seed_ranked = [cell for cell, _ in sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)]
        if seed_weight_sum > 0:
            seed_top_n = int(settings.get(pairwise_name, {}).get("pairwise_seed_top_n", len(seed_ranked)))
            seed_ranked = seed_ranked[: max(1, seed_top_n)]
        else:
            seed_ranked = []
        pairwise_module = modules[pairwise_name]
        if hasattr(pairwise_module, "set_seed_ranked_candidates"):
            pairwise_module.set_seed_ranked_candidates(seed_ranked)
        result = pairwise_module.score(board, candidate_cells, target_number)
        explanations.append(result.explanation)
        normalized = _normalize_scores(result.scores, mode=normalization_mode)
        for c in candidates:
            cell = c["cell"]
            module_score = float(normalized.get(cell, 0.0))
            c["module_scores"][pairwise_name] = module_score
            if result.details:
                c["module_details"][pairwise_name] = result.details.get(cell, {})
            else:
                c["module_details"][pairwise_name] = {}
            if not isinstance(c["module_details"][pairwise_name], dict):
                c["module_details"][pairwise_name] = {}
            if "zone_type" not in c["module_details"][pairwise_name]:
                fair = with_fairness_diagnostics(
                    board,
                    cell,
                    raw_score=float(result.scores.get(cell, 0.5)),
                    bias_corrected_score=module_score,
                    local_radius=1,
                )
                c["module_details"][pairwise_name].update(fair)
            c.setdefault("module_informative", {})
            c["module_informative"][pairwise_name] = _get_informative_value(result, cell)
            c["score"] += module_score * float(weights[pairwise_name])

    return candidates, weights, explanations


def _apply_stage2_adjustment_signals(
    board: List[List[int]],
    candidates: List[Dict[str, object]],
    target_number: int,
    module_settings: Optional[Dict[str, Dict[str, object]]] = None,
    stage1_weights: Optional[Dict[str, float]] = None,
) -> None:
    if not candidates:
        return
    settings = module_settings or {}
    stage2_cfg = dict(settings.get("stage2_adjustments", {}))
    if not stage2_cfg:
        stage2_cfg = {
            "global_assignment_enabled": True,
            "pairwise_enabled": True,
            "assignment_delta_scale": 0.08,
            "assignment_penalty_scale": 0.08,
            "pairwise_delta_scale": 0.06,
            "pairwise_penalty_scale": 0.06,
        }
    candidate_cells = [c["cell"] for c in candidates]
    for cand in candidates:
        cand["assignment_delta"] = 0.0
        cand["assignment_penalty"] = 0.0
        cand["pairwise_delta"] = 0.0
        cand["pairwise_penalty"] = 0.0
        cand["assignment_diagnostics"] = {}
        cand["pairwise_diagnostics"] = {}

    if bool(stage2_cfg.get("global_assignment_enabled", True)):
        assign_cfg = dict(settings.get("global_assignment_prior", {}))
        assign_module = GlobalAssignmentPriorModule(
            assignment_mode=str(assign_cfg.get("assignment_mode", "greedy")),
            top_m_candidates=int(assign_cfg.get("top_m_candidates", 4)),
            exact_max_candidates=int(assign_cfg.get("exact_max_candidates", 20)),
        )
        assign_res = assign_module.score(board, candidate_cells, target_number)
        delta_scale = float(stage2_cfg.get("assignment_delta_scale", 0.08))
        penalty_scale = float(stage2_cfg.get("assignment_penalty_scale", 0.08))
        for cand in candidates:
            cell = cand["cell"]
            s = float(assign_res.scores.get(cell, 0.5))
            cand["assignment_delta"] = max(0.0, s - 0.5) * delta_scale
            cand["assignment_penalty"] = max(0.0, 0.5 - s) * penalty_scale
            cand["assignment_diagnostics"] = assign_res.details.get(cell, {}) if assign_res.details else {}

    if bool(stage2_cfg.get("pairwise_enabled", True)):
        pair_cfg = dict(settings.get("pairwise_conditional_consistency", {}))
        pair_module = PairwiseConditionalConsistencyModule(
            anchor_top_k_cells=int(pair_cfg.get("anchor_top_k_cells", 5)),
            anchor_top_k_values=int(pair_cfg.get("anchor_top_k_values", 5)),
            max_pair_trials_per_candidate=int(pair_cfg.get("max_pair_trials_per_candidate", 20)),
            gating_enabled=bool(pair_cfg.get("gating_enabled", True)),
            contradiction_penalty_weight=float(pair_cfg.get("contradiction_penalty_weight", 1.0)),
            hard_violation_threshold=float(pair_cfg.get("hard_violation_threshold", 2.0)),
            hard_gate_multiplier=float(pair_cfg.get("hard_gate_multiplier", 0.05)),
            soft_gate_floor=float(pair_cfg.get("soft_gate_floor", 0.25)),
            runtime_mode=str(pair_cfg.get("runtime_mode", "fast")),
            candidate_top_n=int(pair_cfg.get("candidate_top_n", 8)),
            global_assignment_mode=str(pair_cfg.get("global_assignment_mode", "greedy")),
            global_assignment_top_m_candidates=int(pair_cfg.get("global_assignment_top_m_candidates", 4)),
        )
        if stage1_weights:
            ranked = sorted(
                candidates,
                key=lambda c: sum(
                    float(c.get("module_scores", {}).get(m, 0.0)) * float(stage1_weights.get(m, 0.0))
                    for m in stage1_weights
                ),
                reverse=True,
            )
            pair_module.set_seed_ranked_candidates([c["cell"] for c in ranked])
        pair_res = pair_module.score(board, candidate_cells, target_number)
        delta_scale = float(stage2_cfg.get("pairwise_delta_scale", 0.06))
        penalty_scale = float(stage2_cfg.get("pairwise_penalty_scale", 0.06))
        for cand in candidates:
            cell = cand["cell"]
            s = float(pair_res.scores.get(cell, 0.5))
            cand["pairwise_delta"] = max(0.0, s - 0.5) * delta_scale
            cand["pairwise_penalty"] = max(0.0, 0.5 - s) * penalty_scale
            cand["pairwise_diagnostics"] = pair_res.details.get(cell, {}) if pair_res.details else {}


def score_candidates(
    board: List[List[int]],
    candidates: List[Dict[str, object]],
    target_number: int,
    module_weights: Optional[Dict[str, float]] = None,
    module_settings: Optional[Dict[str, Dict[str, object]]] = None,
    normalization_mode: str = "disabled",
) -> Tuple[List[Dict[str, object]], Dict[str, float], List[str]]:
    return score_candidates_raw(
        board=board,
        candidates=candidates,
        target_number=target_number,
        module_weights=module_weights,
        module_settings=module_settings,
        normalization_mode=normalization_mode,
    )


def rank_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if candidates and "final_rank_position" in candidates[0]:
        return sorted(candidates, key=lambda item: int(item.get("final_rank_position", 10**9)))
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _score_sorted(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def _safe_top_m(value: object, default: int = 5) -> int:
    try:
        top_m = int(value)
    except (TypeError, ValueError):
        top_m = default
    return max(3, min(10, top_m))


def _distance_penalty_weight(anchor: Cell, candidate: Cell, metric: str, d1: float, d2: float) -> float:
    manhattan = abs(anchor[0] - candidate[0]) + abs(anchor[1] - candidate[1])
    chebyshev = max(abs(anchor[0] - candidate[0]), abs(anchor[1] - candidate[1]))
    if metric == "manhattan":
        if manhattan == 1:
            return d1
        if manhattan == 2:
            return d2
        return 0.0
    if metric == "chebyshev":
        if chebyshev == 1:
            return d1
        if chebyshev == 2:
            return d2
        return 0.0
    # hybrid: use chebyshev for immediate neighborhood, then manhattan for distance-2 ring.
    if chebyshev == 1:
        return d1
    if manhattan == 2:
        return d2
    return 0.0


def _apply_spatial_cluster_penalty(
    ranked: List[Dict[str, object]],
    spatial_cfg: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    enabled = bool(spatial_cfg.get("enabled", False))
    metric = str(spatial_cfg.get("distance_metric", "hybrid")).lower()
    if metric not in {"manhattan", "chebyshev", "hybrid"}:
        metric = "hybrid"
    top_m = _safe_top_m(spatial_cfg.get("top_m", 5), default=5)
    if not enabled or not ranked:
        return ranked, {
            "enabled": enabled,
            "applied": False,
            "distance_metric": metric,
            "top_m": top_m,
            "affected_count": 0,
            "total_penalty": 0.0,
        }

    d1 = max(0.0, float(spatial_cfg.get("penalty_d1", 0.10)))
    d2 = max(0.0, float(spatial_cfg.get("penalty_d2", 0.04)))
    score_gap_gate = max(0.0, float(spatial_cfg.get("score_gap_gate", 0.06)))
    protect_sensitive_threshold = float(spatial_cfg.get("protect_target_sensitive_threshold", 0.65))
    protect_structure_threshold = float(spatial_cfg.get("protect_structure_threshold", 0.62))
    protect_adjustment_threshold = float(spatial_cfg.get("protect_adjustment_threshold", 0.03))
    protect_multiplier = _clip(float(spatial_cfg.get("protect_multiplier", 0.5)), lo=0.1, hi=1.0)
    max_penalty_per_candidate = _clip(float(spatial_cfg.get("max_penalty_per_candidate", 0.08)), lo=0.0, hi=0.2)

    top_limit = min(top_m, len(ranked))
    updated = list(ranked)
    affected = 0
    total_penalty = 0.0
    for idx in range(top_limit):
        cand = updated[idx]
        cand["spatial_cluster_penalty"] = 0.0
        cand["spatial_cluster_penalty_sources"] = []

    for idx in range(1, top_limit):
        cand = updated[idx]
        cand_score = float(cand.get("score", 0.0))
        per_candidate_penalty = 0.0
        sources: List[Dict[str, object]] = []
        for aid in range(idx):
            anchor = updated[aid]
            anchor_score = float(anchor.get("score", 0.0))
            score_gap = anchor_score - cand_score
            if score_gap < 0.0 or score_gap > score_gap_gate:
                continue
            weight = _distance_penalty_weight(
                anchor=anchor["cell"],
                candidate=cand["cell"],
                metric=metric,
                d1=d1,
                d2=d2,
            )
            if weight <= 0.0:
                continue
            near_gap_boost = 1.0 - _clip(score_gap / max(score_gap_gate, 1e-9))
            penalty = weight * near_gap_boost
            target_sensitive_score = float(cand.get("target_sensitive_score", cand_score))
            module_scores = cand.get("module_scores", {}) if isinstance(cand.get("module_scores", {}), dict) else {}
            structural_score = max(
                float(module_scores.get("structural_consistency", 0.0)),
                float(module_scores.get("directional_consistency", 0.0)),
                float(module_scores.get("line_consistency", 0.0)),
            )
            positive_adjustment = max(0.0, float(cand.get("assignment_delta", 0.0))) + max(
                0.0, float(cand.get("pairwise_delta", 0.0))
            )
            if (
                target_sensitive_score >= protect_sensitive_threshold
                or structural_score >= protect_structure_threshold
                or positive_adjustment >= protect_adjustment_threshold
            ):
                penalty *= protect_multiplier
            penalty = min(penalty, max_penalty_per_candidate - per_candidate_penalty)
            if penalty <= 0.0:
                continue
            per_candidate_penalty += penalty
            sources.append(
                {
                    "anchor_cell": anchor["cell"],
                    "score_gap": round(score_gap, 6),
                    "penalty": round(penalty, 6),
                }
            )
            if per_candidate_penalty >= max_penalty_per_candidate:
                break
        if per_candidate_penalty <= 0.0:
            continue
        affected += 1
        total_penalty += per_candidate_penalty
        cand["spatial_cluster_penalty"] = round(per_candidate_penalty, 6)
        cand["spatial_cluster_penalty_sources"] = sources
        new_score = cand_score - per_candidate_penalty
        cand["score"] = new_score
        cand["final_score"] = new_score
        cand["ranking_score"] = new_score

    resorted = _score_sorted(updated)
    for pos, cand in enumerate(resorted, start=1):
        cand["final_rank_position"] = pos
    return resorted, {
        "enabled": True,
        "applied": bool(affected > 0),
        "distance_metric": metric,
        "top_m": top_m,
        "affected_count": affected,
        "total_penalty": round(total_penalty, 6),
    }


def _refresh_distribution_diagnostics(
    diagnostics: Dict[str, Any],
    ranked: List[Dict[str, object]],
) -> Dict[str, Any]:
    if not ranked:
        return diagnostics
    final_scores = [float(c.get("score", 0.0)) for c in ranked]
    raw_mean = sum(final_scores) / len(final_scores)
    raw_var = sum((s - raw_mean) ** 2 for s in final_scores) / len(final_scores)
    final_std = math.sqrt(raw_var)
    top_sorted = sorted(final_scores, reverse=True)
    top1_top2_margin = 0.0 if len(top_sorted) < 2 else top_sorted[0] - top_sorted[1]
    topk = top_sorted[: min(5, len(top_sorted))]
    top1_top5_mean_gap = 0.0 if not topk else top_sorted[0] - (sum(topk) / len(topk))
    tau = max(0.05, final_std)
    exp_values = [math.exp((s - top_sorted[0]) / tau) for s in top_sorted]
    z = sum(exp_values) or 1.0
    probs = [x / z for x in exp_values]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    max_entropy = math.log(max(len(probs), 1))
    diagnostics.update(
        {
            "raw_score_min": min(final_scores),
            "raw_score_max": max(final_scores),
            "raw_score_std": final_std,
            "final_score_min": min(final_scores),
            "final_score_max": max(final_scores),
            "final_score_std": final_std,
            "top1_top2_margin": top1_top2_margin,
            "top1_top5_mean_gap": top1_top5_mean_gap,
            "score_entropy_like": entropy / max(max_entropy, 1e-12),
            "collapsed_score_flag": final_std < 0.02 or top1_top2_margin < 0.01,
            "final_top1_cell": ranked[0]["cell"],
            "top1_changed_by_tiebreak": bool(
                diagnostics.get("stage_a_top1_cell") is not None
                and diagnostics.get("stage_a_top1_cell") != ranked[0]["cell"]
            ),
        }
    )
    return diagnostics


def map_score_to_confidence_1_100(
    margin_to_top2: float,
    effective_candidate_count: int,
    gated_candidate_count: int,
) -> float:
    if effective_candidate_count <= 1:
        return 99.0
    margin_factor = _clip(margin_to_top2 / 0.25)
    density_factor = 1.0 - _clip((gated_candidate_count - 1) / max(effective_candidate_count - 1, 1))
    raw = 35.0 + 45.0 * margin_factor + 20.0 * density_factor
    return round(_clip(raw, lo=1.0, hi=99.0), 2)


def _extract_contradiction_penalty(module_name: str, module_score: float, details: Dict[str, object]) -> float:
    if module_name == "logic_rule":
        return float(details.get("local_contradiction_penalty", 0.0))
    if module_name == "directional_consistency":
        return float(details.get("directional_contradiction_penalty", 0.0))
    if module_name == "line_consistency":
        return float(details.get("line_contradiction_penalty", 0.0))
    if module_name == "global_assignment_prior":
        return float(details.get("anchor_cost_delta_vs_best", 0.0)) + 0.5 * float(
            details.get("infeasible_or_high_cost_flag", 0.0)
        )
    return _clip(1.0 - module_score)


def collect_module_outputs(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    aggregator_cfg: Dict[str, object],
) -> Dict[str, object]:
    module_names = sorted(weights.keys())
    normalization = str(aggregator_cfg.get("competitor_normalization", "per_module_minmax"))
    cells = [c["cell"] for c in candidates]
    module_raw: Dict[str, Dict[Cell, float]] = {}
    module_norm: Dict[str, Dict[Cell, float]] = {}
    module_rank: Dict[str, Dict[Cell, int]] = {}
    module_vote: Dict[str, Dict[Cell, Dict[str, float]]] = {}
    module_informative: Dict[str, Dict[Cell, float]] = {}
    for name in module_names:
        raw = {c["cell"]: float(c.get("module_scores", {}).get(name, 0.0)) for c in candidates}
        informative = {c["cell"]: float(c.get("module_informative", {}).get(name, 1.0)) for c in candidates}
        norm = normalize_scores_per_module(raw, mode=normalization)
        rank = compute_dense_ranks(norm)
        vote = compute_vote_signals(rank)
        module_raw[name] = raw
        module_norm[name] = norm
        module_rank[name] = rank
        module_vote[name] = vote
        module_informative[name] = informative
    stage_a_score_by_cell = {
        cell: sum(float(module_norm[m][cell]) for m in module_names) / max(len(module_names), 1)
        for cell in cells
    }
    stage_a_rank_by_cell = compute_dense_ranks(stage_a_score_by_cell)
    stage_a_top1_cell = max(stage_a_score_by_cell, key=stage_a_score_by_cell.get)
    return {
        "cells": cells,
        "module_names": module_names,
        "module_raw_score_by_cell": module_raw,
        "module_norm_score_by_cell": module_norm,
        "module_rank_by_cell": module_rank,
        "module_vote_by_cell": module_vote,
        "module_informative_by_cell": module_informative,
        "stage_a_score_by_cell": stage_a_score_by_cell,
        "stage_a_rank_by_cell": stage_a_rank_by_cell,
        "stage_a_top1_cell": stage_a_top1_cell,
        "stage_a_top1_score": float(stage_a_score_by_cell[stage_a_top1_cell]),
    }


def build_competitive_fusion_features(
    candidates: List[Dict[str, object]],
    stage_a: Dict[str, object],
    aggregator_cfg: Dict[str, object],
) -> None:
    include_votes = bool(aggregator_cfg.get("include_vote_features", True))
    include_ranks = bool(aggregator_cfg.get("include_rank_features", True))
    include_scores = bool(aggregator_cfg.get("include_score_features", True))
    module_names: List[str] = list(stage_a["module_names"])
    module_norm: Dict[str, Dict[Cell, float]] = stage_a["module_norm_score_by_cell"]
    module_rank: Dict[str, Dict[Cell, int]] = stage_a["module_rank_by_cell"]
    module_vote: Dict[str, Dict[Cell, Dict[str, float]]] = stage_a["module_vote_by_cell"]
    vote_agg = aggregate_topk_votes(module_vote, stage_a["cells"])
    borda = borda_scores(module_rank, stage_a["cells"])
    rrf = rrf_scores(module_rank, stage_a["cells"], k=float(aggregator_cfg.get("vote_rrf_k", 10.0)))
    for c in candidates:
        cell = c["cell"]
        scores = [float(module_norm[m][cell]) for m in module_names]
        ranks = [int(module_rank[m][cell]) for m in module_names]
        mean_score = sum(scores) / max(len(scores), 1)
        var_score = sum((s - mean_score) ** 2 for s in scores) / max(len(scores), 1)
        c["mean_score"] = mean_score
        c["std_score"] = math.sqrt(var_score)
        c["score_spread"] = max(scores) - min(scores) if scores else 0.0
        c["borda_score"] = float(borda[cell])
        c["rrf_score"] = float(rrf[cell])
        c["top1_vote_count"] = float(vote_agg[cell]["top1_vote_count"])
        c["top3_vote_count"] = float(vote_agg[cell]["top3_vote_count"])
        c["top5_vote_count"] = float(vote_agg[cell]["top5_vote_count"])
        c["disagreement_count"] = float(sum(1 for r in ranks if r > 3))
        c["rank_entropy_like"] = float(compute_rank_entropy_like(ranks))
        c["support_margin_to_next"] = float(
            stage_a["stage_a_top1_score"] - float(stage_a["stage_a_score_by_cell"][cell])
        )
        c["conflict_mass"] = float(sum(abs(s - mean_score) for s in scores) / max(len(scores), 1))
        if include_scores or include_ranks or include_votes:
            for m in module_names:
                if include_scores:
                    c[f"module_{m}_score"] = float(module_norm[m][cell])
                if include_ranks:
                    c[f"module_{m}_rank"] = int(module_rank[m][cell])
                if include_votes:
                    c[f"module_{m}_is_top1"] = float(module_vote[m][cell]["is_top1"])
                    c[f"module_{m}_is_top3"] = float(module_vote[m][cell]["is_top3"])
                    c[f"module_{m}_is_top5"] = float(module_vote[m][cell]["is_top5"])
                details_obj = c.get("module_details", {}).get(m, {})
                details = details_obj if isinstance(details_obj, dict) else {}
                c[f"module_{m}_contradiction_penalty"] = float(
                    _extract_contradiction_penalty(m, float(c.get("module_scores", {}).get(m, 0.0)), details)
                )
                c[f"module_{m}_gate_multiplier"] = 1.0


def apply_weighted_rank_fusion(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    stage_a: Dict[str, object],
    aggregator_cfg: Dict[str, object],
) -> None:
    use_borda = str(aggregator_cfg.get("rank_fusion_method", "rrf")) == "borda"
    module_rank: Dict[str, Dict[Cell, int]] = stage_a["module_rank_by_cell"]
    cells: List[Cell] = stage_a["cells"]
    rank_component = borda_scores(module_rank, cells) if use_borda else rrf_scores(module_rank, cells)
    contradiction_weight = float(aggregator_cfg.get("contradiction_penalty_weight", 1.0))
    for c in candidates:
        cell = c["cell"]
        module_details = c.get("module_details", {})
        contradiction = 0.0
        weighted = 0.0
        for name, weight in weights.items():
            details_obj = module_details.get(name, {})
            details = details_obj if isinstance(details_obj, dict) else {}
            contradiction += (
                _extract_contradiction_penalty(name, float(c["module_scores"].get(name, 0.0)), details) * weight
            )
            weighted += weight
        contradiction_penalty = contradiction / max(weighted, 1e-12)
        c["contradiction_penalty"] = contradiction_penalty
        c["gate_multiplier"] = 1.0
        c["vote_bonus"] = float(c.get("top1_vote_count", 0.0))
        c["gated_score"] = float(rank_component[cell])
        c["score"] = float(rank_component[cell] - contradiction_weight * contradiction_penalty)


def apply_vote_fusion(
    candidates: List[Dict[str, object]],
    stage_a: Dict[str, object],
    aggregator_cfg: Dict[str, object],
) -> None:
    alpha1 = float(aggregator_cfg.get("vote_top1_weight", 1.0))
    alpha3 = float(aggregator_cfg.get("vote_top3_weight", 0.7))
    alpha5 = float(aggregator_cfg.get("vote_top5_weight", 0.5))
    for c in candidates:
        vote_score = (
            alpha1 * float(c.get("top1_vote_count", 0.0))
            + alpha3 * float(c.get("top3_vote_count", 0.0))
            + alpha5 * float(c.get("top5_vote_count", 0.0))
        )
        c["contradiction_penalty"] = float(c.get("conflict_mass", 0.0))
        c["gate_multiplier"] = 1.0
        c["vote_bonus"] = vote_score
        c["gated_score"] = vote_score
        c["score"] = vote_score


def apply_meta_judge(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    stage_a: Dict[str, object],
    aggregator_cfg: Dict[str, object],
    use_for_primary_ranking: bool = True,
) -> Optional[str]:
    artifact_path = str(aggregator_cfg.get("judge_artifact_path", "artifacts/competitive_judge_artifact.json"))
    expected_schema_version = str(aggregator_cfg.get("judge_feature_schema_version", "competitive_features_v1"))
    fallback_mode = str(aggregator_cfg.get("fallback_mode", "weighted_rank_fusion"))
    try:
        artifact = load_meta_judge_artifact(artifact_path)
        validate_meta_judge_artifact(artifact)
        artifact_schema = str(artifact.get("schema_version", ""))
        if artifact_schema != expected_schema_version:
            raise ValueError(
                f"judge schema mismatch: artifact={artifact_schema}, expected={expected_schema_version}"
            )
    except (FileNotFoundError, ValueError) as exc:
        if not use_for_primary_ranking:
            return f"meta_judge_fallback:{exc}"
        if fallback_mode == "weighted_rank_fusion":
            apply_weighted_rank_fusion(candidates, weights, stage_a, aggregator_cfg)
        elif fallback_mode == "vote_based_fusion":
            apply_vote_fusion(candidates, stage_a, aggregator_cfg)
        else:
            raise InferenceError(f"Unsupported fallback_mode: {fallback_mode}") from exc
        return f"meta_judge_fallback:{exc}"
    feature_names = list(artifact.get("feature_names", []))
    if not feature_names:
        raise InferenceError("meta judge artifact has empty feature_names")
    for c in candidates:
        feature_row = build_meta_judge_feature_row(c, feature_names)
        score = score_with_logistic_artifact(feature_row, artifact)
        c["contradiction_penalty"] = float(c.get("conflict_mass", 0.0))
        c["gate_multiplier"] = 1.0
        c["vote_bonus"] = float(c.get("top1_vote_count", 0.0))
        c["gated_score"] = score
        c["judge_score"] = score
        if use_for_primary_ranking:
            c["score"] = score
        c["judge_artifact_version"] = str(artifact.get("created_at", artifact.get("schema_version", "unknown")))
        c["judge_feature_schema_version"] = str(artifact.get("schema_version", "unknown"))
    return None


def _normalize_active_weights(
    module_names: List[str],
    module_informative: Dict[str, float],
    yaml_weights: Dict[str, float],
    weighting_mode: str,
) -> Dict[str, float]:
    active = [m for m in module_names if float(module_informative.get(m, 0.0)) > 0.0]
    if not active:
        return {}
    if weighting_mode == "equal_informative":
        each = 1.0 / len(active)
        return {m: each for m in active}
    if weighting_mode == "yaml_normalized":
        base_sum = sum(float(yaml_weights.get(m, 0.0)) for m in active)
        if base_sum <= 0:
            each = 1.0 / len(active)
            return {m: each for m in active}
        return {m: float(yaml_weights.get(m, 0.0)) / base_sum for m in active}
    raise InferenceError(f"Unsupported weighting_mode: {weighting_mode}")


def apply_committee_weighted_sum(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    aggregator_cfg: Dict[str, object],
) -> Dict[str, object]:
    if not candidates:
        return {}
    module_names = sorted(weights.keys())
    weighting_mode = str(aggregator_cfg.get("weighting_mode", "equal_informative"))
    per_module_active_counts = {m: 0 for m in module_names}
    no_info_any = False
    for cand in candidates:
        informative_map = {
            m: float(cand.get("module_informative", {}).get(m, 1.0))
            for m in module_names
        }
        active_weights = _normalize_active_weights(
            module_names=module_names,
            module_informative=informative_map,
            yaml_weights=weights,
            weighting_mode=weighting_mode,
        )
        for m in active_weights:
            per_module_active_counts[m] += 1
        if not active_weights:
            committee_score = 0.5
            no_info = True
            no_info_any = True
            assignment_delta = 0.0
            assignment_penalty = 0.0
            pairwise_delta = 0.0
            pairwise_penalty = 0.0
        else:
            committee_score = sum(
                float(cand.get("module_scores", {}).get(m, 0.0)) * float(active_weights[m]) for m in active_weights
            )
            no_info = False
            assignment_delta = float(cand.get("assignment_delta", 0.0))
            assignment_penalty = float(cand.get("assignment_penalty", 0.0))
            pairwise_delta = float(cand.get("pairwise_delta", 0.0))
            pairwise_penalty = float(cand.get("pairwise_penalty", 0.0))
        stage1_base = float(committee_score)
        stage2_score = stage1_base + assignment_delta - assignment_penalty
        stage3_score = stage2_score + pairwise_delta - pairwise_penalty
        cand["stage1_base_score"] = stage1_base
        cand["stage2_assignment_adjusted_score"] = stage2_score
        cand["stage3_pairwise_adjusted_score"] = stage3_score
        cand["committee_score"] = stage1_base
        cand["final_score"] = stage3_score
        cand["score"] = stage3_score
        cand["ranking_score"] = stage3_score
        cand["active_module_count"] = int(len(active_weights))
        cand["active_weight_sum"] = float(sum(active_weights.values()))
        cand["module_effective_weights"] = dict(active_weights)
        cand["committee_weighting_mode"] = weighting_mode
        cand["top_decision_source"] = "stage3_pairwise_adjusted_score"
        cand["score_chain"] = {
            "stage1_base_score": stage1_base,
            "assignment_delta": assignment_delta,
            "assignment_penalty": assignment_penalty,
            "stage2_assignment_adjusted_score": stage2_score,
            "pairwise_delta": pairwise_delta,
            "pairwise_penalty": pairwise_penalty,
            "stage3_pairwise_adjusted_score": stage3_score,
            "final_score": stage3_score,
        }
        cand["no_informative_modules"] = bool(no_info)
    return {
        "fusion_mode": "committee_weighted_sum",
        "ranking_contract_version": "committee_weighted_sum_v1",
        "committee_weighting_mode": weighting_mode,
        "no_informative_modules": bool(no_info_any),
        "active_module_names": [m for m, n in per_module_active_counts.items() if n > 0],
        "inactive_module_names": [m for m, n in per_module_active_counts.items() if n == 0],
        "per_module_participation_rate": {
            m: (float(per_module_active_counts[m]) / max(len(candidates), 1)) for m in module_names
        },
        "final_top_determined_by": "score_chain_final_score",
    }


def finalize_candidate_ranking(
    candidates: List[Dict[str, object]],
    stage_a: Dict[str, object],
) -> List[Dict[str, object]]:
    final_order = sorted(
        candidates,
        key=lambda x: (
            float(x.get("final_score", x.get("score", 0.0))),
            float(x.get("target_sensitive_score", stage_a["stage_a_score_by_cell"].get(x["cell"], 0.0))),
            float(x.get("support_score", stage_a["stage_a_score_by_cell"].get(x["cell"], 0.0))),
        ),
        reverse=True,
    )
    for idx, cand in enumerate(final_order, start=1):
        cand["stage_a_rank"] = int(stage_a["stage_a_rank_by_cell"].get(cand["cell"], idx))
        cand["final_rank_position"] = int(idx)
        cand["was_reordered_by_tiebreak"] = bool(cand["stage_a_rank"] != idx)
        cand["primary_locked_top1"] = False
        cand["stage_a_margin_to_top1"] = float(
            stage_a["stage_a_top1_score"] - float(stage_a["stage_a_score_by_cell"].get(cand["cell"], 0.0))
        )
        cand["final_score"] = float(cand["score"])
        cand["ranking_score"] = float(cand["score"])
        cand["support_score"] = float(stage_a["stage_a_score_by_cell"].get(cand["cell"], 0.0))
        cand["target_sensitive_score"] = float(stage_a["stage_a_score_by_cell"].get(cand["cell"], 0.0))
        cand["target_agnostic_score"] = float(cand.get("mean_score", 0.0))
        cand["target_sensitivity_gap"] = float(cand["target_sensitive_score"] - cand["target_agnostic_score"])
        cand["target_primary_score"] = float(cand["target_sensitive_score"])
    return final_order


def aggregate_candidate_scores(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    aggregator_cfg: Dict[str, object],
) -> Dict[str, float]:
    agg_type = str(aggregator_cfg.get("type", "competitive_ensemble"))
    if agg_type == "committee_weighted_sum":
        if not candidates:
            return {}
        stage_a = collect_module_outputs(candidates, weights, aggregator_cfg)
        build_competitive_fusion_features(
            candidates,
            stage_a,
            dict(
                aggregator_cfg,
                include_vote_features=bool(aggregator_cfg.get("diagnostics", {}).get("include_vote_features", True)),
                include_rank_features=bool(aggregator_cfg.get("diagnostics", {}).get("include_rank_features", True)),
                include_score_features=bool(aggregator_cfg.get("diagnostics", {}).get("include_score_features", True)),
            ),
        )
        diag = apply_committee_weighted_sum(candidates, weights, aggregator_cfg)
        judge_cfg = dict(aggregator_cfg.get("judge", {}))
        fallback_reason = None
        if bool(judge_cfg.get("enabled", False)):
            fallback_reason = apply_meta_judge(
                candidates,
                weights,
                stage_a,
                {
                    **aggregator_cfg,
                    "judge_artifact_path": str(judge_cfg.get("artifact_path", "")),
                    "fallback_mode": "weighted_rank_fusion",
                },
                use_for_primary_ranking=bool(judge_cfg.get("use_for_primary_ranking", False)),
            )
        final_order = finalize_candidate_ranking(candidates, stage_a)
        final_scores = [float(c["score"]) for c in final_order]
        raw_mean = sum(final_scores) / len(final_scores)
        raw_var = sum((s - raw_mean) ** 2 for s in final_scores) / len(final_scores)
        final_std = math.sqrt(raw_var)
        top_sorted = sorted(final_scores, reverse=True)
        top1_top2_margin = 0.0 if len(top_sorted) < 2 else top_sorted[0] - top_sorted[1]
        topk = top_sorted[: min(5, len(top_sorted))]
        top1_top5_mean_gap = 0.0 if not topk else top_sorted[0] - (sum(topk) / len(topk))
        tau = max(0.05, final_std)
        exp_values = [math.exp((s - top_sorted[0]) / tau) for s in top_sorted]
        z = sum(exp_values) or 1.0
        probs = [x / z for x in exp_values]
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        max_entropy = math.log(max(len(probs), 1))
        return {
            "raw_score_min": min(final_scores),
            "raw_score_max": max(final_scores),
            "raw_score_std": final_std,
            "final_score_min": min(final_scores),
            "final_score_max": max(final_scores),
            "final_score_std": final_std,
            "top1_top2_margin": top1_top2_margin,
            "top1_top5_mean_gap": top1_top5_mean_gap,
            "score_entropy_like": entropy / max(max_entropy, 1e-12),
            "collapsed_score_flag": final_std < 0.02 or top1_top2_margin < 0.01,
            "stage_a_top1_cell": stage_a["stage_a_top1_cell"],
            "final_top1_cell": final_order[0]["cell"],
            "top1_changed_by_tiebreak": bool(stage_a["stage_a_top1_cell"] != final_order[0]["cell"]),
            "fallback_reason": fallback_reason,
            **diag,
        }
    if agg_type != "competitive_ensemble":
        return _aggregate_candidate_scores_legacy(
            candidates=candidates,
            weights=weights,
            agg_type=agg_type,
            gating_enabled=bool(aggregator_cfg.get("gating_enabled", True)),
            contradiction_weight=float(aggregator_cfg.get("contradiction_penalty_weight", 1.0)),
            hard_violation_threshold=float(aggregator_cfg.get("hard_violation_threshold", 2.0)),
            hard_gate_multiplier=float(aggregator_cfg.get("hard_gate_multiplier", 0.05)),
            soft_gate_floor=float(aggregator_cfg.get("soft_gate_floor", 0.25)),
            spread_enabled=bool(aggregator_cfg.get("score_spread_enabled", True)),
            spread_temperature=float(aggregator_cfg.get("score_spread_temperature", 0.2)),
        )
    if not candidates:
        return {}
    stage_a = collect_module_outputs(candidates, weights, aggregator_cfg)
    build_competitive_fusion_features(candidates, stage_a, aggregator_cfg)
    fusion_mode = str(aggregator_cfg.get("fusion_mode", "weighted_rank_fusion"))
    if fusion_mode == "weighted_rank_fusion":
        apply_weighted_rank_fusion(candidates, weights, stage_a, aggregator_cfg)
        fallback_reason = None
    elif fusion_mode == "vote_based_fusion":
        apply_vote_fusion(candidates, stage_a, aggregator_cfg)
        fallback_reason = None
    elif fusion_mode == "learned_meta_ranker":
        fallback_reason = apply_meta_judge(candidates, weights, stage_a, aggregator_cfg)
    else:
        raise InferenceError(f"Unsupported competitive fusion mode: {fusion_mode}")

    final_order = finalize_candidate_ranking(candidates, stage_a)

    final_scores = [float(c["score"]) for c in final_order]
    raw_mean = sum(final_scores) / len(final_scores)
    raw_var = sum((s - raw_mean) ** 2 for s in final_scores) / len(final_scores)
    final_std = math.sqrt(raw_var)
    top_sorted = sorted(final_scores, reverse=True)
    top1_top2_margin = 0.0 if len(top_sorted) < 2 else top_sorted[0] - top_sorted[1]
    topk = top_sorted[: min(5, len(top_sorted))]
    top1_top5_mean_gap = 0.0 if not topk else top_sorted[0] - (sum(topk) / len(topk))

    tau = max(0.05, final_std)
    exp_values = [math.exp((s - top_sorted[0]) / tau) for s in top_sorted]
    z = sum(exp_values) or 1.0
    probs = [x / z for x in exp_values]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    max_entropy = math.log(max(len(probs), 1))
    entropy_like = entropy / max(max_entropy, 1e-12)
    collapsed = final_std < 0.02 or top1_top2_margin < 0.01
    return {
        "raw_score_min": min(final_scores),
        "raw_score_max": max(final_scores),
        "raw_score_std": final_std,
        "final_score_min": min(final_scores),
        "final_score_max": max(final_scores),
        "final_score_std": final_std,
        "top1_top2_margin": top1_top2_margin,
        "top1_top5_mean_gap": top1_top5_mean_gap,
        "score_entropy_like": entropy_like,
        "collapsed_score_flag": collapsed,
        "fusion_mode": fusion_mode,
        "ranking_contract_version": "competitive_ensemble_v1",
        "stage_a_top1_cell": stage_a["stage_a_top1_cell"],
        "final_top1_cell": final_order[0]["cell"],
        "top1_changed_by_tiebreak": bool(stage_a["stage_a_top1_cell"] != final_order[0]["cell"]),
        "primary_locked_top1": False,
        "stage_a_competitor_count": len(stage_a["module_names"]),
        "fallback_reason": fallback_reason,
        "judge_artifact_version": str(final_order[0].get("judge_artifact_version", "unknown")),
        "judge_feature_schema_version": str(final_order[0].get("judge_feature_schema_version", "unknown")),
    }


def _aggregate_candidate_scores_legacy(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    agg_type: str,
    gating_enabled: bool,
    contradiction_weight: float,
    hard_violation_threshold: float,
    hard_gate_multiplier: float,
    soft_gate_floor: float,
    spread_enabled: bool,
    spread_temperature: float,
) -> Dict[str, float]:
    ranking_scores: List[float] = []
    for c in candidates:
        module_scores = c["module_scores"]
        module_details = c.get("module_details", {})
        support_fusion = sum(float(module_scores.get(name, 0.0)) * weight for name, weight in weights.items())
        contradiction = 0.0
        weighted = 0.0
        for name, weight in weights.items():
            details = module_details.get(name, {}) if isinstance(module_details.get(name, {}), dict) else {}
            contradiction += _extract_contradiction_penalty(name, float(module_scores.get(name, 0.0)), details) * weight
            weighted += weight
        contradiction_penalty = contradiction / max(weighted, 1e-12)
        row_v = float(module_details.get("directional_consistency", {}).get("row_violation_count", 0.0))
        col_v = float(module_details.get("directional_consistency", {}).get("col_violation_count", 0.0))
        diag_v = float(module_details.get("line_consistency", {}).get("diag_violation_count", 0.0))
        line_flags = (
            float(module_details.get("line_consistency", {}).get("monotonic_break_flag", 0.0))
            + float(module_details.get("line_consistency", {}).get("percentile_outlier_flag", 0.0))
            + float(module_details.get("line_consistency", {}).get("gap_outlier_flag", 0.0))
        )
        violation_score = row_v + col_v + diag_v + line_flags
        gate_multiplier = 1.0
        if gating_enabled:
            if violation_score >= hard_violation_threshold:
                gate_multiplier = hard_gate_multiplier
            else:
                gate_multiplier = max(soft_gate_floor, 1.0 - 0.25 * contradiction_penalty)
        if agg_type == "weighted_average":
            ranking_score = support_fusion
            gated_score = support_fusion
        else:
            gated_score = gate_multiplier * support_fusion
            ranking_score = gated_score - contradiction_weight * contradiction_penalty
        c["support_score"] = support_fusion
        c["contradiction_penalty"] = contradiction_penalty
        c["gated_score"] = gated_score
        c["gate_multiplier"] = gate_multiplier
        c["ranking_score"] = ranking_score
        ranking_scores.append(ranking_score)
    if not candidates:
        return {}
    raw_mean = sum(ranking_scores) / len(ranking_scores)
    raw_var = sum((s - raw_mean) ** 2 for s in ranking_scores) / len(ranking_scores)
    raw_std = math.sqrt(raw_var)
    spread_factor = 1.0
    if spread_enabled:
        spread_factor = 1.0 + min(2.0, spread_temperature / max(raw_std, 0.03))
    final_scores: List[float] = []
    for c in candidates:
        rs = float(c.get("ranking_score", 0.0))
        c["score"] = raw_mean + (rs - raw_mean) * spread_factor if spread_enabled else rs
        final_scores.append(float(c["score"]))
    final_mean = sum(final_scores) / len(final_scores)
    final_var = sum((s - final_mean) ** 2 for s in final_scores) / len(final_scores)
    final_std = math.sqrt(final_var)
    top_sorted = sorted(final_scores, reverse=True)
    top1_top2_margin = 0.0 if len(top_sorted) < 2 else top_sorted[0] - top_sorted[1]
    topk = top_sorted[: min(5, len(top_sorted))]
    top1_top5_mean_gap = 0.0 if not topk else top_sorted[0] - (sum(topk) / len(topk))
    tau = max(0.05, final_std)
    exp_values = [math.exp((s - top_sorted[0]) / tau) for s in top_sorted]
    z = sum(exp_values) or 1.0
    probs = [x / z for x in exp_values]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    max_entropy = math.log(max(len(probs), 1))
    return {
        "raw_score_min": min(ranking_scores),
        "raw_score_max": max(ranking_scores),
        "raw_score_std": raw_std,
        "final_score_min": min(final_scores),
        "final_score_max": max(final_scores),
        "final_score_std": final_std,
        "top1_top2_margin": top1_top2_margin,
        "top1_top5_mean_gap": top1_top5_mean_gap,
        "score_entropy_like": entropy / max(max_entropy, 1e-12),
        "collapsed_score_flag": final_std < 0.02 or top1_top2_margin < 0.01,
        "fusion_mode": "legacy_weighted_baseline",
    }


def _compute_vote_scores(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    aggregator_cfg: Dict[str, object],
    vote_include_modules: Optional[List[str]] = None,
) -> Dict[Cell, float]:
    if not candidates:
        return {}
    module_names = [m for m in (vote_include_modules or list(weights.keys())) if m in weights]
    top1_w = float(aggregator_cfg.get("vote_top1_weight", 1.0))
    top3_w = float(aggregator_cfg.get("vote_top3_weight", 0.7))
    rrf_w = float(aggregator_cfg.get("vote_rrf_weight", 0.8))
    borda_w = float(aggregator_cfg.get("vote_borda_weight", 0.6))
    k_rrf = float(aggregator_cfg.get("vote_rrf_k", 10.0))
    cells = [c["cell"] for c in candidates]
    vote_totals: Dict[Cell, float] = {cell: 0.0 for cell in cells}
    for m in module_names:
        ranked = sorted(candidates, key=lambda x: float(x["module_scores"].get(m, 0.0)), reverse=True)
        n = len(ranked)
        for idx, cand in enumerate(ranked):
            cell = cand["cell"]
            rank = idx + 1
            vote = 0.0
            if rank == 1:
                vote += top1_w
            if rank <= 3:
                vote += top3_w / 3.0
            vote += rrf_w * (1.0 / (k_rrf + rank))
            vote += borda_w * ((n - idx) / max(n, 1))
            vote_totals[cell] += vote
    values = list(vote_totals.values())
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-9:
        return {k: 0.5 for k in vote_totals}
    return {k: (v - min_v) / (max_v - min_v) for k, v in vote_totals.items()}


def _candidate_confidence_1_to_100(
    candidate_score: float,
    top_score: float,
    best_confidence: float,
    score_std: float,
    gate_multiplier: float,
    contradiction_penalty: float,
) -> float:
    if score_std <= 1e-9:
        gap_factor = 1.0 if abs(candidate_score - top_score) < 1e-9 else 0.0
    else:
        rel_gap = max(0.0, top_score - candidate_score) / max(score_std, 0.01)
        gap_factor = math.exp(-rel_gap)
    confidence = best_confidence * gap_factor
    confidence *= max(0.2, min(1.0, gate_multiplier))
    confidence *= max(0.2, 1.0 - 0.4 * contradiction_penalty)
    return round(_clip(confidence, 1.0, 99.0), 2)


def map_best_confidence_1_100(
    margin_to_top2: float,
    top1_top5_mean_gap: float,
    effective_candidate_count: int,
    gated_candidate_count: int,
    score_entropy_like: float,
    collapsed_score_flag: bool,
) -> float:
    if effective_candidate_count <= 1:
        return 99.0
    margin_factor = _clip(margin_to_top2 / 0.2)
    topk_gap_factor = _clip(top1_top5_mean_gap / 0.2)
    density_factor = 1.0 - _clip((gated_candidate_count - 1) / max(effective_candidate_count - 1, 1))
    concentration = 1.0 - _clip(score_entropy_like)
    collapse_penalty = 0.2 if collapsed_score_flag else 0.0
    raw = 20.0 + 35.0 * margin_factor + 20.0 * topk_gap_factor + 15.0 * concentration + 10.0 * density_factor
    raw *= 1.0 - collapse_penalty
    return round(_clip(raw, 1.0, 99.0), 2)


def build_explanation(
    rows: int,
    cols: int,
    target_number: int,
    unopened_count: int,
    module_weights: Dict[str, float],
    best_cell: Optional[Tuple[int, int]],
    module_explanations: List[str],
) -> List[str]:
    reasoning = [
        f"盤面總格數為 {rows * cols}，因此合法數字集合為 1..{rows * cols}",
        f"target_number={target_number} 尚未出現在已開格",
        f"目前共有 {unopened_count} 個未開格",
        "模組加權為 " + ", ".join(f"{k}={v:.2f}" for k, v in module_weights.items()),
    ]
    reasoning.extend(module_explanations)
    if best_cell is not None:
        reasoning.append(f"綜合模組分數後，row={best_cell[0] + 1}, col={best_cell[1] + 1} 最高")
    return reasoning


def _run_inference_detailed(
    board: List[List[int]],
    target_number: int,
    source: str,
    module_weights: Optional[Dict[str, float]] = None,
    module_settings: Optional[Dict[str, Dict[str, object]]] = None,
    version: str = "v1",
    apply_reranker_stage: bool = True,
    include_module_details: bool = True,
    aggregator_config: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    parsed = parse_board_input(board)
    remaining = compute_remaining_numbers(parsed)
    status, opened_cell = validate_target_number(target_number, parsed, remaining)

    unopened_cells_payload = [{"row": r + 1, "col": c + 1} for r, c in parsed.unopened_cells]

    if status == "already_opened" and opened_cell is not None:
        return {
            "status": "already_opened",
            "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
            "target_number": target_number,
            "remaining_numbers": remaining,
            "unopened_cells": unopened_cells_payload,
            "best_cell": {
                "row": opened_cell[0] + 1,
                "col": opened_cell[1] + 1,
                "score": 1.0,
                "confidence_1_to_100": 100.0,
            },
            "candidate_cells": [],
            "confidence_score": 1.0,
            "best_ranking_score": 1.0,
            "best_confidence_score": 1.0,
            "reasoning": [
                f"盤面總格數為 {parsed.rows * parsed.cols}，合法數字集合為 1..{parsed.rows * parsed.cols}",
                f"target_number={target_number} 已經在已開格",
            ],
            "module_contributions": {},
            "metadata": {
                "score_type": "ranking_score",
                "confidence_type": "deterministic_when_already_opened",
                "confidence_1_to_100_type": "fixed_100_for_already_opened",
                "confidence_1_to_100_is_probability": False,
                "score_can_be_negative": False,
                "confidence_score_is_not_ranking_score": True,
                "source": source,
                "version": version,
                "ranking_stage": "baseline_only",
                "reranker_version": None,
                "reranker_feature_schema_version": None,
                "reranker_fallback_reason": "already_opened",
            },
        }

    if not parsed.unopened_cells:
        raise InferenceError("board has no unopened cells")

    candidates = build_cell_candidates(parsed.unopened_cells)
    aggregator_cfg = aggregator_config or load_aggregator_config()
    scored, weights, module_explanations = score_candidates(
        board,
        candidates,
        target_number,
        module_weights=module_weights,
        module_settings=module_settings,
        normalization_mode=str(aggregator_cfg.get("normalization_mode", "disabled")),
    )
    if str(aggregator_cfg.get("type", "")) == "committee_weighted_sum":
        _validate_committee_stage1_modules(weights)
    _apply_stage2_adjustment_signals(
        board=board,
        candidates=scored,
        target_number=target_number,
        module_settings=module_settings,
        stage1_weights=weights,
    )
    diagnostics = aggregate_candidate_scores(scored, weights, aggregator_cfg)
    ranked = rank_candidates(scored)
    spatial_cfg = dict(aggregator_cfg.get("spatial_postprocess", {}))
    ranked, spatial_diag = _apply_spatial_cluster_penalty(ranked, spatial_cfg)
    diagnostics = _refresh_distribution_diagnostics(diagnostics, ranked)
    diagnostics["spatial_postprocess"] = dict(spatial_diag)
    best = ranked[0]

    margin_to_top2 = 0.0 if len(ranked) <= 1 else max(
        0.0,
        float(ranked[0]["score"]) - float(ranked[1]["score"]),
    )
    effective_candidate_count = len(ranked)
    gated_candidate_count = sum(1 for c in ranked if float(c.get("gate_multiplier", 1.0)) > 0.2)
    best_confidence_1_to_100 = map_best_confidence_1_100(
        margin_to_top2,
        float(diagnostics.get("top1_top5_mean_gap", 0.0)),
        effective_candidate_count,
        gated_candidate_count,
        float(diagnostics.get("score_entropy_like", 1.0)),
        bool(diagnostics.get("collapsed_score_flag", 0.0)),
    )
    confidence_reason = (
        "strong_elimination_after_gating"
        if margin_to_top2 >= 0.15 and gated_candidate_count <= max(1, effective_candidate_count // 2)
        else "limited_elimination_power"
    )

    sensitive_ranked = sorted(
        ranked,
        key=lambda x: float(x.get("target_sensitive_score", 0.0)),
        reverse=True,
    )
    sensitive_rank_map = {item["cell"]: i + 1 for i, item in enumerate(sensitive_ranked)}

    candidate_cells = []
    preserve_diagnostics = bool(aggregator_cfg.get("preserve_diagnostics", True))
    for idx, cell in enumerate(ranked, start=1):
        score = round(float(cell["score"]), 6)
        cell_profile = support_profile(board, cell["cell"], local_radius=1)
        final_confidence = round(
            max(
                0.0,
                min(
                    1.0,
                    0.65 * float(cell_profile["coverage_ratio"])
                    + 0.35
                    * float(
                        sum(float(v) for v in cell.get("module_informative", {}).values())
                        / max(len(cell.get("module_informative", {})), 1)
                    ),
                ),
            ),
            6,
        )
        edge_bias_adjustment = round(
            float(max(0.0, 0.5 - cell_profile["coverage_ratio"]))
            if cell_profile["zone_type"] in {"corner", "edge"}
            else 0.0,
            6,
        )
        board_coverage_adjustment = round(float(cell_profile["global_support"] - 0.5), 6)
        payload = {
                "row": cell["cell"][0] + 1,
                "col": cell["cell"][1] + 1,
                "score": score,
                "confidence_1_to_100": _candidate_confidence_1_to_100(
                    candidate_score=score,
                    top_score=float(ranked[0]["score"]),
                    best_confidence=best_confidence_1_to_100,
                    score_std=float(diagnostics.get("final_score_std", 0.0)),
                    gate_multiplier=float(cell.get("gate_multiplier", 1.0)),
                    contradiction_penalty=float(cell.get("contradiction_penalty", 0.0)),
                ),
                "module_scores": {
                    k: round(float(v), 6) for k, v in sorted(cell["module_scores"].items())
                },
                "primitive_module_scores": {
                    k: round(float(v), 6) for k, v in sorted(cell["module_scores"].items())
                },
                "module_informative": {
                    k: round(float(v), 6) for k, v in sorted(cell.get("module_informative", {}).items())
                },
                "module_confidences": {
                    k: round(float(v), 6) for k, v in sorted(cell.get("module_informative", {}).items())
                },
                "module_effective_weights": {
                    k: round(float(v), 6) for k, v in sorted(cell.get("module_effective_weights", {}).items())
                },
                "support_score": round(float(cell.get("support_score", score)), 6),
                "contradiction_penalty": round(float(cell.get("contradiction_penalty", 0.0)), 6),
                "gated_score": round(float(cell.get("gated_score", score)), 6),
                "ranking_score": round(float(cell.get("ranking_score", score)), 6),
                "final_score": score,
                "final_confidence": final_confidence,
                "committee_score": round(float(cell.get("committee_score", score)), 6),
                "stage1_base_score": round(float(cell.get("stage1_base_score", score)), 6),
                "assignment_delta": round(float(cell.get("assignment_delta", 0.0)), 6),
                "assignment_penalty": round(float(cell.get("assignment_penalty", 0.0)), 6),
                "pairwise_delta": round(float(cell.get("pairwise_delta", 0.0)), 6),
                "pairwise_penalty": round(float(cell.get("pairwise_penalty", 0.0)), 6),
                "score_chain": cell.get("score_chain", {}),
                "active_module_count": int(cell.get("active_module_count", len(weights))),
                "active_weight_sum": round(float(cell.get("active_weight_sum", 1.0)), 6),
                "committee_weighting_mode": str(cell.get("committee_weighting_mode", "")),
                "top_decision_source": str(cell.get("top_decision_source", "")),
                "gate_multiplier": round(float(cell.get("gate_multiplier", 1.0)), 6),
                "vote_bonus": round(float(cell.get("vote_bonus", 0.0)), 6),
                "target_sensitive_score": round(float(cell.get("target_sensitive_score", 0.0)), 6),
                "target_agnostic_score": round(float(cell.get("target_agnostic_score", 0.0)), 6),
                "target_sensitivity_gap": round(float(cell.get("target_sensitivity_gap", 0.0)), 6),
                "target_primary_score": round(float(cell.get("target_primary_score", 0.0)), 6),
                "target_sensitive_rank": int(sensitive_rank_map.get(cell["cell"], idx)),
                "final_rank": int(idx),
                "stage_a_rank": int(cell.get("stage_a_rank", sensitive_rank_map.get(cell["cell"], idx))),
                "stage_a_margin_to_top1": round(float(cell.get("stage_a_margin_to_top1", 0.0)), 6),
                "was_reordered_by_tiebreak": bool(cell.get("was_reordered_by_tiebreak", False)),
                "primary_locked_top1": bool(cell.get("primary_locked_top1", False)),
                "tie_break_score": round(float(cell.get("tie_break_score", 0.0)), 6),
                "module_details": cell.get("module_details", {}) if include_module_details else {},
                "zone_type": cell_profile["zone_type"],
                "edge_bias_adjustment": edge_bias_adjustment,
                "board_coverage_adjustment": board_coverage_adjustment,
                "spatial_cluster_penalty": round(float(cell.get("spatial_cluster_penalty", 0.0)), 6),
                "spatial_cluster_penalty_sources": cell.get("spatial_cluster_penalty_sources", []),
        }
        if preserve_diagnostics:
            for key in (
                "mean_score",
                "std_score",
                "score_spread",
                "top1_vote_count",
                "top3_vote_count",
                "top5_vote_count",
                "borda_score",
                "rrf_score",
                "disagreement_count",
                "rank_entropy_like",
                "support_margin_to_next",
                "conflict_mass",
            ):
                if key in cell:
                    payload[key] = round(float(cell[key]), 6)
            for key, value in cell.items():
                if key.startswith("module_") and isinstance(value, (int, float)):
                    payload[key] = round(float(value), 6)
        candidate_cells.append(payload)

    reasoning = build_explanation(
        parsed.rows,
        parsed.cols,
        target_number,
        len(parsed.unopened_cells),
        weights,
        best["cell"],
        module_explanations,
    )

    baseline_candidate_cells = list(candidate_cells)
    runtime_mode = str(load_fast_path_config().get("runtime_mode", "fast"))
    rerank_meta = {
        "ranking_stage": "baseline_only",
        "reranker_version": None,
        "reranker_feature_schema_version": None,
        "reranker_fallback_reason": "reranker_disabled_by_runtime_flag",
    }
    if apply_reranker_stage:
        artifact, reason = load_reranker_artifact()
        if artifact is None or not artifact.get("enabled", False):
            rerank_meta = {
                "ranking_stage": "baseline_only",
                "reranker_version": None if artifact is None else artifact.get("version"),
                "reranker_feature_schema_version": None if artifact is None else artifact.get("feature_schema_version"),
                "reranker_fallback_reason": reason or "reranker_disabled",
            }
        else:
            feature_rows = build_candidate_feature_rows(
                case_id=f"runtime:{target_number}",
                board_shape=(parsed.rows, parsed.cols),
                candidates=baseline_candidate_cells,
                true_cell_1_based=None,
                board=board,
                target_number=target_number,
            )
            candidate_cells, rerank_meta = apply_reranker(baseline_candidate_cells, feature_rows)

    best_cell_payload = candidate_cells[0]
    best_score = round(float(best_cell_payload["score"]), 6)
    best_confidence_score = round(best_confidence_1_to_100 / 100.0, 6)
    return {
        "status": "ok",
        "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
        "target_number": target_number,
        "remaining_numbers": remaining,
        "unopened_cells": unopened_cells_payload,
        "best_cell": {
            "row": best_cell_payload["row"],
            "col": best_cell_payload["col"],
            "score": best_score,
            "confidence_1_to_100": best_confidence_1_to_100,
        },
        "candidate_cells": candidate_cells,
        "confidence_score": best_confidence_score,
        "best_ranking_score": best_score,
        "best_confidence_score": best_confidence_score,
        "reasoning": reasoning,
        "module_contributions": weights,
        "metadata": {
            "score_type": "ranking_score",
            "confidence_type": "margin_and_elimination_aware",
            "confidence_1_to_100_type": "gap_density_mapping_non_calibrated",
            "confidence_1_to_100_is_probability": False,
            "best_cell_confidence_1_to_100": best_confidence_1_to_100,
            "score_can_be_negative": True,
            "confidence_score_is_not_ranking_score": True,
            "margin_to_top2": round(float(margin_to_top2), 6),
            "effective_candidate_count": effective_candidate_count,
            "gated_candidate_count": gated_candidate_count,
            "confidence_reason": confidence_reason,
            **{
                k: (
                    round(v, 6)
                    if isinstance(v, float)
                    else (bool(v) if isinstance(v, bool) else v)
                )
                for k, v in diagnostics.items()
            },
            "source": source,
            "version": version,
            "aggregation_type": str(aggregator_cfg.get("type", "weighted_average")),
            "fusion_mode": str(aggregator_cfg.get("fusion_mode", "weighted_rank_fusion")),
            "committee_weighting_mode": str(diagnostics.get("committee_weighting_mode", "")),
            "no_informative_modules": bool(diagnostics.get("no_informative_modules", False)),
            "active_module_names": diagnostics.get("active_module_names", []),
            "inactive_module_names": diagnostics.get("inactive_module_names", []),
            "per_module_participation_rate": diagnostics.get("per_module_participation_rate", {}),
            "final_top_determined_by": str(diagnostics.get("final_top_determined_by", "")),
            "judge_model_type": str(aggregator_cfg.get("judge_model_type", "")),
            "judge_artifact_path": str(
                aggregator_cfg.get("judge_artifact_path", aggregator_cfg.get("judge", {}).get("artifact_path", ""))
            ),
            "judge_artifact_version": str(diagnostics.get("judge_artifact_version", "unknown")),
            "judge_feature_schema_version": str(diagnostics.get("judge_feature_schema_version", "unknown")),
            "equal_start_enabled": bool(aggregator_cfg.get("type", "") == "competitive_ensemble"),
            "legacy_weighted_path_used": bool(aggregator_cfg.get("type", "") != "competitive_ensemble"),
            "normalization_mode": str(
                aggregator_cfg.get("competitor_normalization", aggregator_cfg.get("normalization_mode", "disabled"))
            ),
            "gating_enabled": bool(aggregator_cfg.get("gating_enabled", False)),
            "runtime_mode": runtime_mode,
            "fallback_reason": diagnostics.get("fallback_reason"),
            "elimination_version": str(aggregator_cfg.get("elimination_version", "v1")),
            "spatial_postprocess_enabled": bool(spatial_diag.get("enabled", False)),
            "spatial_postprocess_applied": bool(spatial_diag.get("applied", False)),
            "spatial_postprocess_top_m": int(spatial_diag.get("top_m", 5)),
            "spatial_postprocess_distance_metric": str(spatial_diag.get("distance_metric", "hybrid")),
            "spatial_postprocess_affected_count": int(spatial_diag.get("affected_count", 0)),
            "spatial_postprocess_total_penalty": round(float(spatial_diag.get("total_penalty", 0.0)), 6),
            "ranking_stage": rerank_meta["ranking_stage"],
            "reranker_version": rerank_meta["reranker_version"],
            "reranker_feature_schema_version": rerank_meta["reranker_feature_schema_version"],
            "reranker_fallback_reason": rerank_meta["reranker_fallback_reason"],
        },
    }


def build_target_cell_score_matrix(
    per_target_results: Dict[int, Dict[str, Any]],
    unopened_cells: List[Cell],
    contradiction_alpha: float,
    gate_beta: float,
) -> Tuple[List[int], List[List[float]], Dict[int, Dict[Cell, Dict[str, Any]]]]:
    target_numbers = sorted(per_target_results.keys())
    per_target_cell_scores: Dict[int, Dict[Cell, Dict[str, Any]]] = {}
    matrix: List[List[float]] = []
    for target in target_numbers:
        row: List[float] = []
        by_cell: Dict[Cell, Dict[str, Any]] = {}
        for cand in per_target_results[target]["candidate_cells"]:
            cell = (int(cand["row"]) - 1, int(cand["col"]) - 1)
            by_cell[cell] = cand
        per_target_cell_scores[target] = by_cell
        for cell in unopened_cells:
            cand = by_cell.get(cell)
            if cand is None:
                row.append(1.0)
                continue
            base_score = float(cand.get("score", 0.0))
            contradiction_penalty = float(cand.get("contradiction_penalty", 0.0))
            gate_multiplier = float(cand.get("gate_multiplier", 1.0))
            gate_bonus = max(0.0, gate_multiplier - 0.5)
            joint_score = _clip(base_score - contradiction_alpha * contradiction_penalty + gate_beta * gate_bonus)
            row.append(1.0 - joint_score)
        matrix.append(row)
    return target_numbers, matrix, per_target_cell_scores


def solve_joint_assignment(
    target_numbers: List[int],
    unopened_cells: List[Cell],
    cost_matrix: List[List[float]],
    assignment_mode: str = "exact",
) -> Tuple[Dict[int, Cell], str]:
    if not target_numbers:
        return {}, assignment_mode
    if len(target_numbers) > len(unopened_cells):
        raise InferenceError("target_numbers exceed unopened cell count")

    if assignment_mode == "exact" and linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        if len(row_idx) == len(target_numbers):
            assigned: Dict[int, Cell] = {}
            for r_idx, c_idx in zip(row_idx.tolist(), col_idx.tolist()):
                assigned[target_numbers[r_idx]] = unopened_cells[c_idx]
            if len(assigned) == len(target_numbers):
                return assigned, "exact"

    used_cells: set[Cell] = set()
    assigned = {}
    ranked_pairs: List[Tuple[float, int, Cell]] = []
    for r_idx, target in enumerate(target_numbers):
        for c_idx, cell in enumerate(unopened_cells):
            ranked_pairs.append((cost_matrix[r_idx][c_idx], target, cell))
    ranked_pairs.sort(key=lambda x: x[0])
    for _, target, cell in ranked_pairs:
        if target in assigned or cell in used_cells:
            continue
        assigned[target] = cell
        used_cells.add(cell)
        if len(assigned) == len(target_numbers):
            break
    if len(assigned) != len(target_numbers):
        raise InferenceError("joint assignment failed to assign all targets uniquely")
    return assigned, "greedy"


def dedup_ranked_candidates(
    per_target_results: Dict[int, Dict[str, Any]],
    assignments: Dict[int, Cell],
    per_target_cell_scores: Dict[int, Dict[Cell, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    output = []
    for target, result in sorted(per_target_results.items()):
        selected = assignments[target]
        selected_cand = per_target_cell_scores[target][selected]
        top1 = result["candidate_cells"][0]
        top1_cell = (int(top1["row"]) - 1, int(top1["col"]) - 1)
        base_score = float(selected_cand.get("score", 0.0))
        top1_score = float(top1.get("score", 0.0))
        output.append(
            {
                "target_number": target,
                "row": selected[0] + 1,
                "col": selected[1] + 1,
                "joint_score": base_score,
                "base_score": base_score,
                "was_reassigned_from_individual_top1": selected != top1_cell,
                "individual_top1_row": int(top1["row"]),
                "individual_top1_col": int(top1["col"]),
                "reassignment_cost_delta": round(max(0.0, top1_score - base_score), 6),
            }
        )
    return output


def run_multi_target_inference(
    board: List[List[int]],
    target_numbers: List[int],
    source: str,
    apply_reranker_stage: bool = False,
) -> Dict[str, Any]:
    if not target_numbers:
        raise InferenceError("target_numbers must be non-empty")
    if len(set(target_numbers)) != len(target_numbers):
        raise InferenceError("target_numbers must be unique")

    parsed = parse_board_input(board)
    remaining = compute_remaining_numbers(parsed)
    for target in target_numbers:
        validate_target_number(target, parsed, remaining)

    joint_cfg = load_joint_assignment_config()
    per_target_results: Dict[int, Dict[str, Any]] = {}
    for target in target_numbers:
        per_target_results[target] = _run_inference_detailed(
            board=board,
            target_number=target,
            source=source,
            apply_reranker_stage=apply_reranker_stage,
        )

    target_order, cost_matrix, per_target_cell_scores = build_target_cell_score_matrix(
        per_target_results=per_target_results,
        unopened_cells=parsed.unopened_cells,
        contradiction_alpha=float(joint_cfg.get("contradiction_alpha", 0.35)),
        gate_beta=float(joint_cfg.get("gate_beta", 0.2)),
    )
    assignments, used_mode = solve_joint_assignment(
        target_numbers=target_order,
        unopened_cells=parsed.unopened_cells,
        cost_matrix=cost_matrix,
        assignment_mode=str(joint_cfg.get("assignment_mode", "exact")),
    )
    deduped = dedup_ranked_candidates(per_target_results, assignments, per_target_cell_scores)

    top1_cells = [
        (res["candidate_cells"][0]["row"], res["candidate_cells"][0]["col"])
        for res in per_target_results.values()
    ]
    duplicate_before = len(top1_cells) - len(set(top1_cells))
    assigned_cells = [(item["row"], item["col"]) for item in deduped]
    duplicate_after = len(assigned_cells) - len(set(assigned_cells))
    per_target_ranked_candidates = {
        str(target): per_target_results[target]["candidate_cells"][
            : int(joint_cfg.get("top_k_per_target", 10))
        ]
        for target in target_order
    }

    return {
        "status": "ok",
        "board_shape": {"rows": parsed.rows, "cols": parsed.cols},
        "target_numbers": target_order,
        "assignments": deduped,
        "assignment_score_table": {
            str(target): {
                f"{cell[0] + 1},{cell[1] + 1}": round(1.0 - cost_matrix[r_idx][c_idx], 6)
                for c_idx, cell in enumerate(parsed.unopened_cells)
            }
            for r_idx, target in enumerate(target_order)
        },
        "per_target_ranked_candidates": per_target_ranked_candidates,
        "metadata": {
            "assignment_mode": used_mode,
            "dedup_enabled": bool(joint_cfg.get("dedup_enabled", True)),
            "duplicate_top1_count_before_assignment": duplicate_before,
            "duplicate_top1_count_after_assignment": duplicate_after,
            "joint_assignment_version": str(joint_cfg.get("version", "v1")),
        },
    }


def compact_top10_response(result: Dict[str, Any]) -> Dict[str, Any]:
    if "candidate_cells" not in result:
        raise InferenceError("missing candidate_cells in inference result")
    if not result["candidate_cells"] and result.get("best_cell"):
        best = result["best_cell"]
        return {
            "top10": [
                {
                    "row": int(best["row"]),
                    "col": int(best["col"]),
                    "confidence_1_to_100": round(float(best.get("confidence_1_to_100", 100.0)), 2),
                }
            ],
            "best_confidence_1_to_100": round(float(best.get("confidence_1_to_100", 100.0)), 2),
        }
    ranked = sorted(
        list(result["candidate_cells"]),
        key=lambda x: float(x.get("confidence_1_to_100", 0.0)),
        reverse=True,
    )
    top10 = []
    for cand in ranked[:10]:
        top10.append(
            {
                "row": int(cand["row"]),
                "col": int(cand["col"]),
                "confidence_1_to_100": round(float(cand["confidence_1_to_100"]), 2),
            }
        )
    if not top10:
        raise InferenceError("empty candidate_cells is not allowed in compact response")
    return {
        "top10": top10,
        "best_confidence_1_to_100": float(top10[0]["confidence_1_to_100"]),
    }


def run_inference(
    board: List[List[int]],
    target_number: int,
    source: str,
    module_weights: Optional[Dict[str, float]] = None,
    module_settings: Optional[Dict[str, Dict[str, object]]] = None,
    version: str = "v1",
    apply_reranker_stage: bool = True,
) -> Dict[str, Any]:
    detailed = _run_inference_detailed(
        board=board,
        target_number=target_number,
        source=source,
        module_weights=module_weights,
        module_settings=module_settings,
        version=version,
        apply_reranker_stage=apply_reranker_stage,
        include_module_details=False,
    )
    return compact_top10_response(detailed)
