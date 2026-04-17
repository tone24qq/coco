# flake8: noqa: F401,E501
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd

from src.inference_config import (
    load_aggregator_config,
    load_fast_path_config,
    load_joint_assignment_config,
    load_module_settings,
    load_module_weights,
    load_trained_ranker_config,
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
from src.main_ranker import MainRankerError, score_candidates_with_ranker
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
    if chebyshev == 1:
        return d1
    if manhattan == 2:
        return d2
    return 0.0


def _candidate_evidence_protection_factor(cand: Dict[str, object], spatial_cfg: Dict[str, object]) -> float:
    protect_sensitive_threshold = float(spatial_cfg.get("protect_target_sensitive_threshold", 0.65))
    protect_structure_threshold = float(spatial_cfg.get("protect_structure_threshold", 0.62))
    protect_adjustment_threshold = float(spatial_cfg.get("protect_adjustment_threshold", 0.03))
    protect_multiplier = _clip(float(spatial_cfg.get("protect_multiplier", 0.5)), lo=0.1, hi=1.0)
    cand_score = float(cand.get("score", 0.0))
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
        return protect_multiplier
    return 1.0



def _apply_spatial_cluster_penalty(
    ranked: List[Dict[str, object]],
    spatial_cfg: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not ranked or not bool(spatial_cfg.get("enabled", False)):
        return ranked, {"applied": False, "method": spatial_cfg.get("method", "spatial_penalty")}
    top_m = _safe_top_m(spatial_cfg.get("top_m", 5))
    method = str(spatial_cfg.get("method", "spatial_penalty"))
    metric = str(spatial_cfg.get("distance_metric", "hybrid"))
    d1 = float(spatial_cfg.get("penalty_d1", 0.1))
    d2 = float(spatial_cfg.get("penalty_d2", 0.04))
    max_penalty = float(spatial_cfg.get("max_penalty_per_candidate", 0.08))

    out = [dict(c) for c in ranked]
    anchors = out[:max(1, min(3, len(out)))]
    for idx, cand in enumerate(out[:top_m]):
        if idx == 0:
            cand["spatial_cluster_penalty"] = 0.0
            continue
        penalty = 0.0
        for a in anchors:
            penalty += _distance_penalty_weight(a["cell"], cand["cell"], metric, d1, d2)
        if method == "evidence_aware_mmr":
            penalty = max(penalty, d2 * 0.5)
            penalty *= _candidate_evidence_protection_factor(cand, spatial_cfg)
        penalty = min(max_penalty, penalty)
        cand["spatial_cluster_penalty"] = float(penalty)
        cand["score"] = float(cand.get("score", 0.0)) - float(penalty)

    out_sorted = sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out_sorted, {"applied": True, "method": method, "top_m": top_m}


def aggregate_candidate_scores(
    candidates: List[Dict[str, object]],
    module_weights: Dict[str, float],
    aggregator_config: Dict[str, object],
) -> Dict[str, object]:
    agg_type = str(aggregator_config.get("type", "committee_weighted_sum"))
    score_values: List[float] = []
    for cand in candidates:
        ms = cand.get("module_scores", {}) if isinstance(cand.get("module_scores"), dict) else {}
        inf = cand.get("module_informative", {}) if isinstance(cand.get("module_informative"), dict) else {}
        informative_active = {m: float(module_weights.get(m, 0.0)) for m in ms if float(inf.get(m, 1.0)) > 0}
        has_inf_map = isinstance(inf, dict) and bool(inf)
        if not informative_active and has_inf_map:
            eff = {}
        else:
            active = informative_active or {m: float(module_weights.get(m, 0.0)) for m in ms}
            wm = str(aggregator_config.get("weighting_mode", "yaml_normalized"))
            if wm == "equal_informative" and active:
                eff = {m: 1.0 / len(active) for m in active}
            else:
                s = sum(max(0.0, w) for w in active.values())
                eff = {m: (max(0.0, w) / s if s > 0 else 1.0 / max(1, len(active))) for m, w in active.items()}
        cand["module_effective_weights"] = eff
        cand["active_module_count"] = len(eff)

        base = 0.5 if not eff else sum(float(ms.get(m, 0.0)) * float(w) for m, w in eff.items())
        penalty = float(cand.get("module_details", {}).get("logic_rule", {}).get("local_contradiction_penalty", 0.0))
        score = base - 0.2 * penalty
        cand["stage1_base_score"] = score
        cand.setdefault("assignment_delta", 0.0)
        cand.setdefault("assignment_penalty", 0.0)
        cand.setdefault("pairwise_delta", 0.0)
        cand.setdefault("pairwise_penalty", 0.0)
        cand["committee_score"] = score + cand["assignment_delta"] - cand["assignment_penalty"] + cand["pairwise_delta"] - cand["pairwise_penalty"]
        cand["score"] = cand["committee_score"]
        cand["score_chain"] = {"stage1_base_score": cand["stage1_base_score"], "stage2_assignment_adjusted_score": cand["stage1_base_score"] + cand["assignment_delta"] - cand["assignment_penalty"], "stage3_pairwise_adjusted_score": cand["committee_score"], "assignment_delta": cand["assignment_delta"], "assignment_penalty": cand["assignment_penalty"], "pairwise_delta": cand["pairwise_delta"], "pairwise_penalty": cand["pairwise_penalty"], "final_score": cand["score"]}
        cand["target_sensitive_score"] = float(ms.get("logic_rule", score))
        cand["support_score"] = float(sum(ms.values()) / max(len(ms), 1)) if ms else score
        score_values.append(score)

    fallback_reason = None
    if agg_type == "competitive_ensemble":
        mode = str(aggregator_config.get("fusion_mode", "weighted_rank_fusion"))
        if mode not in {"weighted_rank_fusion", "vote_based_fusion", "learned_meta_ranker"}:
            raise InferenceError("invalid competitive fusion_mode")
        for cand in candidates:
            cand["mean_score"] = float(cand.get("support_score", 0.0))
            cand["rrf_score"] = float(cand.get("score", 0.0))
            cand["top1_vote_count"] = 0
            for m, val in (cand.get("module_scores", {}) or {}).items():
                cand[f"module_{m}_rank"] = 1
                cand[f"module_{m}_is_top3"] = int(float(val) >= 0.5)
                cand[f"module_{m}_is_top1"] = int(float(val) >= 0.9)
            cand["stage_a_rank"] = 1
        if mode == "learned_meta_ranker":
            artifact = Path(str(aggregator_config.get("judge_artifact_path", "")))
            if not artifact.exists():
                fallback_reason = "meta_judge_fallback_missing_artifact"
                for cand in candidates:
                    cand["fallback_reason"] = fallback_reason

    if str(aggregator_config.get("type", "")) == "gate_then_weighted_sum" and bool(aggregator_config.get("gating_enabled", False)):
        thr = float(aggregator_config.get("hard_violation_threshold", 2.0))
        for cand in candidates:
            d = cand.get("module_details", {}) if isinstance(cand.get("module_details", {}), dict) else {}
            dc = d.get("directional_consistency", {}) if isinstance(d.get("directional_consistency", {}), dict) else {}
            lc = d.get("line_consistency", {}) if isinstance(d.get("line_consistency", {}), dict) else {}
            vio = float(dc.get("row_violation_count", 0.0)) + float(dc.get("col_violation_count", 0.0)) + float(lc.get("monotonic_break_flag", 0.0)) + float(lc.get("percentile_outlier_flag", 0.0))
            if vio >= thr:
                cand["score"] = cand["score"] * 0.05
    if bool(aggregator_config.get("score_spread_enabled", False)) and candidates:
        vals = [float(c.get("score", 0.0)) for c in candidates]
        mean_v = sum(vals) / len(vals)
        for cand in candidates:
            cand["score"] = mean_v + 2.0 * (float(cand.get("score", 0.0)) - mean_v)

    spatial_cfg = aggregator_config.get("spatial_postprocess", {}) if isinstance(aggregator_config.get("spatial_postprocess", {}), dict) else {}
    spatial_diag = {"applied": False, "method": "spatial_penalty", "top_m": _safe_top_m(5)}
    if spatial_cfg.get("enabled", False):
        sorted_now = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        sorted_now, spatial_diag = _apply_spatial_cluster_penalty(sorted_now, spatial_cfg)
        candidates[:] = sorted_now

    ordered = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for i, cand in enumerate(ordered, start=1):
        cand["final_rank_position"] = i
        cand["top_decision_source"] = "stage3_pairwise_adjusted_score"
        cand["was_reordered_by_tiebreak"] = False

    raw_std = float(pd.Series(score_values).std(ddof=0)) if score_values else 0.0
    final_std = float(pd.Series([float(c.get("score", 0.0)) for c in candidates]).std(ddof=0)) if candidates else 0.0
    return {
        "aggregation_type": agg_type,
        "committee_weighting_mode": str(aggregator_config.get("weighting_mode", "yaml_normalized")),
        "fusion_mode": str(aggregator_config.get("fusion_mode", "weighted_rank_fusion")),
        "raw_score_std": raw_std,
        "final_score_std": final_std,
        "collapsed_score_flag": bool(final_std < 1e-12),
        "spatial_postprocess_applied": bool(spatial_diag.get("applied", False)),
        "spatial_postprocess_method": str(spatial_diag.get("method", "spatial_penalty")),
        "spatial_postprocess_top_m": int(spatial_diag.get("top_m", _safe_top_m(5))),
        "fallback_reason": fallback_reason,
        "no_informative_modules": bool(all(int(c.get("active_module_count", 0)) == 0 for c in candidates)),
    }


def solve_joint_assignment(
    target_numbers: List[int],
    unopened_cells: List[Cell],
    cost_matrix: List[List[float]],
    assignment_mode: str = "exact",
) -> Tuple[Dict[int, Cell], str]:
    if assignment_mode == "exact" and linear_sum_assignment is not None:
        import numpy as np

        arr = np.array(cost_matrix, dtype=float)
        rows, cols = linear_sum_assignment(arr)
        return {int(target_numbers[r]): unopened_cells[int(c)] for r, c in zip(rows, cols)}, "exact"

    assigned: Dict[int, Cell] = {}
    used: set[Cell] = set()
    for i, t in enumerate(target_numbers):
        best_cell = None
        best_cost = float("inf")
        for j, cell in enumerate(unopened_cells):
            if cell in used:
                continue
            c = float(cost_matrix[i][j])
            if c < best_cost:
                best_cost = c
                best_cell = cell
        if best_cell is None:
            continue
        assigned[int(t)] = best_cell
        used.add(best_cell)
    return assigned, "greedy"


def run_multi_target_inference(
    board: List[List[int]],
    target_numbers: List[int],
    source: str,
) -> Dict[str, Any]:
    per_target = [_run_inference_detailed(board, t, source=source, apply_reranker_stage=False) for t in target_numbers]
    unopened = [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == -1]
    cost_matrix: List[List[float]] = []
    duplicate_before = 0
    top1_cells: List[Tuple[int, int]] = []
    for out in per_target:
        cells = out.get("candidate_cells", [])
        if cells:
            top1_cells.append((int(cells[0]["row"]) - 1, int(cells[0]["col"]) - 1))
        row_costs = []
        for cell in unopened:
            found = next((c for c in cells if int(c["row"]) - 1 == cell[0] and int(c["col"]) - 1 == cell[1]), None)
            score = float(found.get("score", 0.0)) if found else -1e9
            row_costs.append(-score)
        cost_matrix.append(row_costs)
    duplicate_before = len(top1_cells) - len(set(top1_cells))

    assignment, mode = solve_joint_assignment(target_numbers, unopened, cost_matrix, assignment_mode="exact")
    assignments = []
    for t in target_numbers:
        cell = assignment[int(t)]
        assignments.append(
            {
                "target_number": int(t),
                "row": int(cell[0] + 1),
                "col": int(cell[1] + 1),
                "was_reassigned_from_individual_top1": (cell not in set(top1_cells)),
            }
        )
    duplicate_after = len(assignments) - len({(a["row"], a["col"]) for a in assignments})
    return {
        "status": "ok",
        "assignments": assignments,
        "metadata": {
            "assignment_mode": mode,
            "duplicate_top1_count_before_assignment": int(duplicate_before),
            "duplicate_top1_count_after_assignment": int(duplicate_after),
        },
    }


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
    if str(aggregator_cfg.get("type", "committee_weighted_sum")) == "committee_weighted_sum":
        _validate_committee_stage1_modules(module_weights or load_module_weights())
    scored, weights, module_explanations = score_candidates(
        board,
        candidates,
        target_number,
        module_weights=module_weights,
        module_settings=module_settings,
        normalization_mode=str(aggregator_cfg.get("normalization_mode", "disabled")),
    )

    agg_diag = aggregate_candidate_scores(scored, weights, aggregator_cfg)
    ranked = rank_candidates(scored)
    best = ranked[0]
    margin_to_top2 = 0.0 if len(ranked) <= 1 else max(0.0, float(ranked[0]["score"]) - float(ranked[1]["score"]))
    effective_candidate_count = len(ranked)
    gated_candidate_count = len(ranked)
    best_confidence_1_to_100 = map_best_confidence_1_100(
        margin_to_top2,
        0.0,
        effective_candidate_count,
        gated_candidate_count,
        0.5,
        False,
    )

    candidate_cells = []
    for idx, cell in enumerate(ranked, start=1):
        score = round(float(cell["score"]), 6)
        cell_profile = support_profile(board, cell["cell"], local_radius=1)
        payload = {
            "row": cell["cell"][0] + 1,
            "col": cell["cell"][1] + 1,
            "score": score,
            "confidence_1_to_100": round(best_confidence_1_to_100 if idx == 1 else max(1.0, best_confidence_1_to_100 - idx * 3.0), 2),
            "module_scores": {k: round(float(v), 6) for k, v in sorted(cell["module_scores"].items())},
            "module_informative": {k: round(float(v), 6) for k, v in sorted(cell.get("module_informative", {}).items())},
            "module_details": cell.get("module_details", {}) if include_module_details else {},
            "zone_type": cell_profile.get("zone_type", "unknown"),
            "final_rank": int(idx),
            "module_effective_weights": cell.get("module_effective_weights", {}),
            "active_module_count": int(cell.get("active_module_count", 0)),
            "stage1_base_score": float(cell.get("stage1_base_score", cell.get("score", 0.0))),
            "assignment_delta": float(cell.get("assignment_delta", 0.0)),
            "assignment_penalty": float(cell.get("assignment_penalty", 0.0)),
            "pairwise_delta": float(cell.get("pairwise_delta", 0.0)),
            "pairwise_penalty": float(cell.get("pairwise_penalty", 0.0)),
            "score_chain": cell.get("score_chain", {}),
            "final_score": float(cell.get("score", 0.0)),
            "spatial_cluster_penalty": float(cell.get("spatial_cluster_penalty", 0.0)),
            "target_sensitive_score": float(cell.get("target_sensitive_score", 0.0)),
            "mean_score": float(cell.get("mean_score", 0.0)),
            "rrf_score": float(cell.get("rrf_score", 0.0)),
            "top1_vote_count": int(cell.get("top1_vote_count", 0)),
        }
        for k, v in cell.items():
            if k.startswith("module_") and (k.endswith("_is_top1") or k.endswith("_is_top3") or k.endswith("_rank")):
                payload[k] = v
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

    trained_ranker_cfg = load_trained_ranker_config()
    if apply_reranker_stage and bool(trained_ranker_cfg.get("enabled", False)):
        try:
            candidate_pairs = [(int(c["row"]) - 1, int(c["col"]) - 1) for c in baseline_candidate_cells]
            scores, model_meta = score_candidates_with_ranker(
                board=board,
                target_number=target_number,
                candidates=candidate_pairs,
                strict_missing_artifact=bool(trained_ranker_cfg.get("strict_missing_artifact", True)),
                registry_path=Path(str(trained_ranker_cfg.get("model_registry_path", "artifacts/model_registry.json"))),
            )
            for cand, score in zip(baseline_candidate_cells, scores):
                cand["trained_ranker_score"] = float(score)
            baseline_candidate_cells.sort(key=lambda x: float(x.get("trained_ranker_score", 0.0)), reverse=True)
            for i, cand in enumerate(baseline_candidate_cells, start=1):
                cand["final_rank"] = i
            rerank_meta = {
                "ranking_stage": "trained_ranker_primary",
                "reranker_version": "main_ranker_v1",
                "reranker_feature_schema_version": "main_ranker_v1",
                "reranker_fallback_reason": None,
                "model_meta": model_meta,
            }
        except MainRankerError as exc:
            if bool(trained_ranker_cfg.get("strict_missing_artifact", False)):
                raise InferenceError(f"trained_ranker_strict_mode: {exc}") from exc
            rerank_meta = {
                "ranking_stage": "baseline_only",
                "reranker_version": None,
                "reranker_feature_schema_version": None,
                "reranker_fallback_reason": f"trained_ranker_missing:{exc}",
                "model_meta": {"model_used": "none"},
            }

    if apply_reranker_stage and bool(trained_ranker_cfg.get("apply_heuristic_rerank_after_model", True)):
        artifact, reason = load_reranker_artifact()
        if artifact is not None and artifact.get("enabled", False):
            feature_rows = build_candidate_feature_rows(
                case_id=f"runtime:{target_number}",
                board_shape=(parsed.rows, parsed.cols),
                candidates=baseline_candidate_cells,
                true_cell_1_based=None,
                board=board,
                target_number=target_number,
            )
            baseline_candidate_cells, rerank_meta = apply_reranker(baseline_candidate_cells, feature_rows)
        elif rerank_meta["ranking_stage"] == "baseline_only":
            rerank_meta = {
                "ranking_stage": "baseline_only",
                "reranker_version": None if artifact is None else artifact.get("version"),
                "reranker_feature_schema_version": None if artifact is None else artifact.get("feature_schema_version"),
                "reranker_fallback_reason": reason or "reranker_disabled",
            }

    candidate_cells = baseline_candidate_cells
    best_cell_payload = candidate_cells[0]
    best_score = round(float(best_cell_payload.get("score", 0.0)), 6)
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
            "source": source,
            "version": version,
            "runtime_mode": runtime_mode,
            "ranking_stage": rerank_meta["ranking_stage"],
            "reranker_version": rerank_meta["reranker_version"],
            "reranker_feature_schema_version": rerank_meta["reranker_feature_schema_version"],
            "reranker_fallback_reason": rerank_meta["reranker_fallback_reason"],
            "model_strategy": rerank_meta.get("model_meta", {}).get("model_strategy"),
            "model_used": rerank_meta.get("model_meta", {}).get("model_used"),
            "size_class": rerank_meta.get("model_meta", {}).get("size_class"),
            "fallback_used": rerank_meta.get("model_meta", {}).get("fallback_used"),
            "fallback_reason": rerank_meta.get("model_meta", {}).get("fallback_reason"),
            "aggregation_type": agg_diag.get("aggregation_type"),
            "fusion_mode": agg_diag.get("fusion_mode"),
            "committee_weighting_mode": agg_diag.get("committee_weighting_mode"),
            "spatial_postprocess_enabled": bool((aggregator_cfg.get("spatial_postprocess", {}) or {}).get("enabled", False)),
            "spatial_postprocess_applied": agg_diag.get("spatial_postprocess_applied", False),
            "spatial_postprocess_method": agg_diag.get("spatial_postprocess_method", "spatial_penalty"),
            "spatial_postprocess_top_m": agg_diag.get("spatial_postprocess_top_m", _safe_top_m(5)),
            "final_top1_cell": (best_cell_payload["row"] - 1, best_cell_payload["col"] - 1),
            "judge_artifact_path": aggregator_cfg.get("judge_artifact_path") or (aggregator_cfg.get("judge", {}) if isinstance(aggregator_cfg.get("judge", {}), dict) else {}).get("artifact_path"),
            "fallback_reason": agg_diag.get("fallback_reason") or rerank_meta.get("model_meta", {}).get("fallback_reason"),
            "no_informative_modules": bool(agg_diag.get("no_informative_modules", False)),
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
