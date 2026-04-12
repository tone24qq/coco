from __future__ import annotations

from src.inference_service import (
    _run_inference_detailed,
    aggregate_candidate_scores,
    build_cell_candidates,
    score_candidates,
)


def _committee_cfg(weighting_mode: str = "equal_informative") -> dict:
    return {
        "type": "committee_weighted_sum",
        "weighting_mode": weighting_mode,
        "allow_abstain": True,
        "preserve_diagnostics": True,
        "tie_break_order": ["target_sensitive_score", "support_score", "row_col_stable"],
        "diagnostics": {
            "include_vote_features": True,
            "include_rank_features": True,
            "include_score_features": True,
        },
        "judge": {
            "enabled": True,
            "use_for_primary_ranking": False,
            "artifact_path": "artifacts/does_not_exist.json",
        },
    }


def test_equal_informative_weighting_gives_equal_share() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "module_scores": {"a": 0.2, "b": 0.6, "c": 0.9},
            "module_details": {},
            "module_informative": {"a": 1.0, "b": 1.0, "c": 1.0},
        }
    ]
    diag = aggregate_candidate_scores(candidates, {"a": 0.9, "b": 0.09, "c": 0.01}, _committee_cfg())
    eff = candidates[0]["module_effective_weights"]
    assert eff == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert diag["committee_weighting_mode"] == "equal_informative"


def test_yaml_normalized_weighting_uses_only_active_modules() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "module_scores": {"a": 0.2, "b": 0.6, "c": 0.9},
            "module_details": {},
            "module_informative": {"a": 1.0, "b": 0.0, "c": 1.0},
        }
    ]
    aggregate_candidate_scores(
        candidates,
        {"a": 0.2, "b": 0.7, "c": 0.1},
        _committee_cfg(weighting_mode="yaml_normalized"),
    )
    eff = candidates[0]["module_effective_weights"]
    assert set(eff.keys()) == {"a", "c"}
    assert abs(eff["a"] - (0.2 / 0.3)) < 1e-9
    assert abs(eff["c"] - (0.1 / 0.3)) < 1e-9


def test_committee_mode_uses_only_module_sum_for_primary_top() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "module_scores": {"m1": 0.9, "m2": 0.4},
            "module_details": {},
            "module_informative": {"m1": 1.0, "m2": 1.0},
        },
        {
            "cell": (0, 1),
            "module_scores": {"m1": 0.8, "m2": 0.8},
            "module_details": {},
            "module_informative": {"m1": 1.0, "m2": 1.0},
        },
    ]
    aggregate_candidate_scores(candidates, {"m1": 0.5, "m2": 0.5}, _committee_cfg())
    ranked = sorted(candidates, key=lambda c: c["final_rank_position"])
    assert ranked[0]["cell"] == (0, 1)
    assert ranked[0]["top_decision_source"] == "stage3_pairwise_adjusted_score"


def test_meta_judge_cannot_override_committee_primary_score() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "module_scores": {"m1": 0.7},
            "module_details": {},
            "module_informative": {"m1": 1.0},
        },
        {
            "cell": (0, 1),
            "module_scores": {"m1": 0.6},
            "module_details": {},
            "module_informative": {"m1": 1.0},
        },
    ]
    aggregate_candidate_scores(candidates, {"m1": 1.0}, _committee_cfg())
    ranked = sorted(candidates, key=lambda c: c["final_rank_position"])
    assert ranked[0]["cell"] == (0, 0)
    assert all(abs(c["score"] - c["committee_score"]) < 1e-9 for c in candidates)


def test_primary_tie_break_is_stable_and_deterministic() -> None:
    candidates = [
        {
            "cell": (0, 0),
            "module_scores": {"m1": 0.5},
            "module_details": {},
            "module_informative": {"m1": 1.0},
            "target_sensitive_score": 0.8,
            "support_score": 0.8,
        },
        {
            "cell": (0, 1),
            "module_scores": {"m1": 0.5},
            "module_details": {},
            "module_informative": {"m1": 1.0},
            "target_sensitive_score": 0.8,
            "support_score": 0.8,
        },
    ]
    aggregate_candidate_scores(candidates, {"m1": 1.0}, _committee_cfg())
    ranked = sorted(candidates, key=lambda c: c["final_rank_position"])
    assert ranked[0]["cell"] == (0, 0)


def test_candidate_payload_contains_participation_fields() -> None:
    out = _run_inference_detailed([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=False)
    cell = out["candidate_cells"][0]
    assert "module_informative" in cell
    assert "module_effective_weights" in cell
    assert "active_module_count" in cell


def test_neighborhood_association_abstains_when_no_seed() -> None:
    out = _run_inference_detailed(
        [[1, -1, 3], [-1, 5, -1]],
        4,
        source="t",
        apply_reranker_stage=False,
        module_weights={"logic_rule": 0.5, "neighborhood_association": 0.5},
        module_settings={
            "neighborhood_association": {
                "enabled_seed_families": ["same_tail"],
                "min_seed_count": 2,
            }
        },
        aggregator_config=_committee_cfg(),
    )
    cand = out["candidate_cells"][0]
    assert cand["module_informative"]["neighborhood_association"] == 0.0


def test_neighborhood_association_affects_score_when_seed_exists() -> None:
    board = [[19, -1, 29], [39, -1, 49]]
    with_module, _, _ = score_candidates(
        board,
        build_cell_candidates([(0, 1), (1, 1)]),
        target_number=59,
        module_weights={"logic_rule": 0.5, "neighborhood_association": 0.5},
    )
    without_module, _, _ = score_candidates(
        board,
        build_cell_candidates([(0, 1), (1, 1)]),
        target_number=59,
        module_weights={"logic_rule": 1.0},
    )
    s1 = {c["cell"]: c["module_scores"]["neighborhood_association"] for c in with_module}
    s2 = {c["cell"]: c["module_scores"].get("neighborhood_association", -1.0) for c in without_module}
    assert s1 != s2


def test_no_duplicate_scoring_family_in_committee() -> None:
    try:
        _run_inference_detailed(
            [[1, -1, 3], [-1, 5, -1]],
            4,
            source="t",
            apply_reranker_stage=False,
            module_weights={"structural_consistency": 0.5, "directional_consistency": 0.5},
            aggregator_config=_committee_cfg(),
        )
    except Exception as exc:
        assert "structural_consistency" in str(exc)
        return
    raise AssertionError("Expected fail-fast for duplicated structural family in committee stage-1")


def test_structural_consistency_replaces_directional_and_line() -> None:
    scored, _, _ = score_candidates(
        [[1, -1, 3], [-1, 5, -1]],
        build_cell_candidates([(0, 1), (1, 0), (1, 2)]),
        target_number=4,
        module_weights={"structural_consistency": 1.0},
    )
    assert "structural_consistency" in scored[0]["module_scores"]
    assert "directional_consistency" not in scored[0]["module_scores"]
    assert "line_consistency" not in scored[0]["module_scores"]


def test_global_assignment_is_stage2_only() -> None:
    out = _run_inference_detailed([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=False)
    cell = out["candidate_cells"][0]
    assert "global_assignment_prior" not in cell["module_scores"]
    assert "assignment_delta" in cell and "assignment_penalty" in cell


def test_pairwise_is_stage2_only() -> None:
    out = _run_inference_detailed([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=False)
    cell = out["candidate_cells"][0]
    assert "pairwise_conditional_consistency" not in cell["module_scores"]
    assert "pairwise_delta" in cell and "pairwise_penalty" in cell


def test_score_chain_contract() -> None:
    out = _run_inference_detailed([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=False)
    cell = out["candidate_cells"][0]
    assert "score_chain" in cell
    chain = cell["score_chain"]
    assert "stage1_base_score" in chain
    assert "stage2_assignment_adjusted_score" in chain
    assert "stage3_pairwise_adjusted_score" in chain
    assert "final_score" in chain


def test_final_score_depends_on_single_stage1_base_path() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    out = _run_inference_detailed(
        board,
        4,
        source="t",
        apply_reranker_stage=False,
        module_weights={"logic_rule": 0.5, "structural_consistency": 0.5},
        aggregator_config=_committee_cfg(),
    )
    cell = out["candidate_cells"][0]
    expected = (
        float(cell["stage1_base_score"])
        + float(cell["assignment_delta"])
        - float(cell["assignment_penalty"])
        + float(cell["pairwise_delta"])
        - float(cell["pairwise_penalty"])
    )
    assert abs(float(cell["final_score"]) - expected) < 1e-6
