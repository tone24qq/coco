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
from src.ranking_features import build_candidate_feature_rows
from src.reranker import apply_reranker, load_reranker_artifact
from src.scoring_modules import Cell, ModuleScoreResult, build_modules
from src.scoring_modules import linear_sum_assignment


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


def score_candidates(
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
            cell = c["cell"]
            module_score = float(normalized.get(cell, 0.0))
            c["module_scores"][module_name] = module_score
            if result.details:
                c["module_details"][module_name] = result.details.get(cell, {})
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
            c["score"] += module_score * float(weights[pairwise_name])

    return candidates, weights, explanations


def rank_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


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


def aggregate_candidate_scores(
    candidates: List[Dict[str, object]],
    weights: Dict[str, float],
    aggregator_cfg: Dict[str, object],
) -> Dict[str, float]:
    agg_type = str(aggregator_cfg.get("type", "weighted_average"))
    gating_enabled = bool(aggregator_cfg.get("gating_enabled", True))
    contradiction_weight = float(aggregator_cfg.get("contradiction_penalty_weight", 1.0))
    hard_violation_threshold = float(aggregator_cfg.get("hard_violation_threshold", 2.0))
    hard_gate_multiplier = float(aggregator_cfg.get("hard_gate_multiplier", 0.05))
    soft_gate_floor = float(aggregator_cfg.get("soft_gate_floor", 0.25))
    spread_enabled = bool(aggregator_cfg.get("score_spread_enabled", True))
    spread_temperature = float(aggregator_cfg.get("score_spread_temperature", 0.2))
    fusion_mode = str(aggregator_cfg.get("fusion_mode", "weighted_only"))
    vote_alpha = float(aggregator_cfg.get("vote_alpha", 0.15))
    sensitive_modules = list(aggregator_cfg.get("target_sensitive_modules", list(weights.keys())))
    agnostic_modules = list(aggregator_cfg.get("target_agnostic_modules", []))
    max_agnostic_share = float(aggregator_cfg.get("max_target_agnostic_weight_share", 0.2))
    vote_include_modules = list(aggregator_cfg.get("vote_include_modules", sensitive_modules))
    known_modules = set(build_modules().keys())
    unknown_vote_modules = [m for m in vote_include_modules if m not in known_modules]
    if unknown_vote_modules:
        raise InferenceError(f"vote_include_modules contain unknown modules: {unknown_vote_modules}")
    ranking_scores: List[float] = []
    vote_scores: Dict[Cell, float] = _compute_vote_scores(candidates, weights, aggregator_cfg, vote_include_modules)

    for c in candidates:
        cell = c["cell"]
        module_scores = c["module_scores"]
        module_details = c.get("module_details", {})
        sensitive_weight = sum(float(weights.get(m, 0.0)) for m in sensitive_modules)
        agnostic_weight = sum(float(weights.get(m, 0.0)) for m in agnostic_modules)
        target_sensitive_score = (
            sum(float(module_scores.get(name, 0.0)) * float(weights.get(name, 0.0)) for name in sensitive_modules)
            / max(sensitive_weight, 1e-12)
        )
        target_agnostic_score_raw = (
            sum(float(module_scores.get(name, 0.0)) * float(weights.get(name, 0.0)) for name in agnostic_modules)
            / max(agnostic_weight, 1e-12)
            if agnostic_weight > 0
            else 0.0
        )
        max_agnostic = max_agnostic_share * max(target_sensitive_score, 1e-6) / max(1.0 - max_agnostic_share, 1e-6)
        target_agnostic_score = min(target_agnostic_score_raw, max_agnostic)
        support_fusion = target_sensitive_score + target_agnostic_score

        contradiction = 0.0
        weighted = 0.0
        for name, weight in weights.items():
            details = module_details.get(name, {}) if isinstance(module_details.get(name, {}), dict) else {}
            p = _extract_contradiction_penalty(name, float(module_scores.get(name, 0.0)), details)
            contradiction += p * weight
            weighted += weight
        contradiction_penalty = contradiction / max(weighted, 1e-12)

        gate_multiplier = 1.0
        row_v = float(module_details.get("directional_consistency", {}).get("row_violation_count", 0.0))
        col_v = float(module_details.get("directional_consistency", {}).get("col_violation_count", 0.0))
        diag_v = float(module_details.get("line_consistency", {}).get("diag_violation_count", 0.0))
        line_flags = (
            float(module_details.get("line_consistency", {}).get("monotonic_break_flag", 0.0))
            + float(module_details.get("line_consistency", {}).get("percentile_outlier_flag", 0.0))
            + float(module_details.get("line_consistency", {}).get("gap_outlier_flag", 0.0))
        )
        violation_score = row_v + col_v + diag_v + line_flags
        if gating_enabled:
            if violation_score >= hard_violation_threshold:
                gate_multiplier = hard_gate_multiplier
            else:
                gate_multiplier = max(soft_gate_floor, 1.0 - 0.25 * contradiction_penalty)

        if agg_type == "weighted_average":
            gated_score = support_fusion
            ranking_score = support_fusion
        else:
            gated_score = gate_multiplier * support_fusion
            ranking_score = gated_score - contradiction_weight * contradiction_penalty
        vote_bonus = float(vote_scores.get(cell, 0.0))
        if fusion_mode == "vote_only":
            ranking_score = vote_bonus
            gated_score = vote_bonus
        elif fusion_mode == "weighted_plus_vote":
            ranking_score = support_fusion + vote_alpha * vote_bonus
            gated_score = ranking_score
        elif fusion_mode == "weighted_plus_vote_with_gate":
            ranking_score = ranking_score + vote_alpha * vote_bonus

        c["support_score"] = support_fusion
        c["target_sensitive_score"] = target_sensitive_score
        c["target_agnostic_score"] = target_agnostic_score
        c["target_sensitivity_gap"] = target_sensitive_score - target_agnostic_score
        c["contradiction_penalty"] = contradiction_penalty
        c["gated_score"] = gated_score
        c["gate_multiplier"] = gate_multiplier
        c["vote_bonus"] = vote_bonus
        c["ranking_score"] = ranking_score
        ranking_scores.append(ranking_score)

    if not candidates:
        return {}

    raw_mean = sum(ranking_scores) / len(ranking_scores)
    raw_var = sum((s - raw_mean) ** 2 for s in ranking_scores) / len(ranking_scores)
    raw_std = math.sqrt(raw_var)
    raw_min = min(ranking_scores)
    raw_max = max(ranking_scores)

    spread_factor = 1.0
    if spread_enabled:
        spread_factor = 1.0 + min(2.0, spread_temperature / max(raw_std, 0.03))

    final_scores: List[float] = []
    for c in candidates:
        ranking_score = float(c.get("ranking_score", 0.0))
        if spread_enabled:
            final_score = raw_mean + (ranking_score - raw_mean) * spread_factor
        else:
            final_score = ranking_score
        c["score"] = final_score
        final_scores.append(final_score)

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
    entropy_like = entropy / max(max_entropy, 1e-12)
    collapsed = final_std < 0.02 or top1_top2_margin < 0.01
    return {
        "raw_score_min": raw_min,
        "raw_score_max": raw_max,
        "raw_score_std": raw_std,
        "final_score_min": min(final_scores),
        "final_score_max": max(final_scores),
        "final_score_std": final_std,
        "top1_top2_margin": top1_top2_margin,
        "top1_top5_mean_gap": top1_top5_mean_gap,
        "score_entropy_like": entropy_like,
        "collapsed_score_flag": collapsed,
        "fusion_mode": fusion_mode,
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
    aggregator_cfg = load_aggregator_config()
    scored, weights, module_explanations = score_candidates(
        board,
        candidates,
        target_number,
        module_weights=module_weights,
        module_settings=module_settings,
        normalization_mode=str(aggregator_cfg.get("normalization_mode", "disabled")),
    )
    diagnostics = aggregate_candidate_scores(scored, weights, aggregator_cfg)
    ranked = rank_candidates(scored)
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
    for idx, cell in enumerate(ranked, start=1):
        score = round(float(cell["score"]), 6)
        candidate_cells.append(
            {
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
                "support_score": round(float(cell.get("support_score", score)), 6),
                "contradiction_penalty": round(float(cell.get("contradiction_penalty", 0.0)), 6),
                "gated_score": round(float(cell.get("gated_score", score)), 6),
                "ranking_score": round(float(cell.get("ranking_score", score)), 6),
                "final_score": score,
                "gate_multiplier": round(float(cell.get("gate_multiplier", 1.0)), 6),
                "vote_bonus": round(float(cell.get("vote_bonus", 0.0)), 6),
                "target_sensitive_score": round(float(cell.get("target_sensitive_score", 0.0)), 6),
                "target_agnostic_score": round(float(cell.get("target_agnostic_score", 0.0)), 6),
                "target_sensitivity_gap": round(float(cell.get("target_sensitivity_gap", 0.0)), 6),
                "target_sensitive_rank": int(sensitive_rank_map.get(cell["cell"], idx)),
                "final_rank": int(idx),
                "module_details": cell.get("module_details", {}) if include_module_details else {},
            }
        )

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
            "normalization_mode": str(aggregator_cfg.get("normalization_mode", "disabled")),
            "gating_enabled": bool(aggregator_cfg.get("gating_enabled", False)),
            "runtime_mode": runtime_mode,
            "target_sensitive_modules": list(aggregator_cfg.get("target_sensitive_modules", [])),
            "target_agnostic_modules": list(aggregator_cfg.get("target_agnostic_modules", [])),
            "vote_include_modules": list(aggregator_cfg.get("vote_include_modules", [])),
            "elimination_version": str(aggregator_cfg.get("elimination_version", "v1")),
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
