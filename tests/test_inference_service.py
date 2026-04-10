from __future__ import annotations

from src.inference_service import _normalize_scores, build_cell_candidates, rank_candidates, score_candidates


def test_module_weights_take_effect() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    candidates = build_cell_candidates([(0, 1), (1, 0), (1, 2)])

    scored_logic, _, _ = score_candidates(
        board,
        [
            {"cell": c["cell"], "score": 0.0, "module_scores": {}}
            for c in candidates
        ],
        target_number=4,
        module_weights={"logic_rule": 1.0},
    )
    scored_prior, _, _ = score_candidates(
        board,
        [
            {"cell": c["cell"], "score": 0.0, "module_scores": {}}
            for c in candidates
        ],
        target_number=4,
        module_weights={"prior_model": 1.0},
    )

    top_logic = rank_candidates(scored_logic)[0]["cell"]
    top_prior = rank_candidates(scored_prior)[0]["cell"]
    assert top_logic != top_prior


def test_module_settings_applied_even_when_module_weights_passed() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    candidates = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    scored, _, _ = score_candidates(
        board,
        candidates,
        target_number=4,
        module_weights={"global_assignment_prior": 1.0},
        module_settings={"global_assignment_prior": {"assignment_mode": "greedy"}},
    )
    assert "global_assignment_prior" in scored[0]["module_details"]
    assert scored[0]["module_details"]["global_assignment_prior"]["global_assignment_mode"] == 0.0


def test_no_global_module_state_leakage_between_calls() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    c1 = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    out1, _, _ = score_candidates(
        board,
        c1,
        target_number=4,
        module_weights={"global_assignment_prior": 1.0},
        module_settings={"global_assignment_prior": {"assignment_mode": "greedy"}},
    )
    c2 = build_cell_candidates([(0, 1), (1, 0), (1, 2)])
    out2, _, _ = score_candidates(
        board,
        c2,
        target_number=4,
        module_weights={"global_assignment_prior": 1.0},
        module_settings={"global_assignment_prior": {"assignment_mode": "exact"}},
    )
    assert out1[0]["module_details"]["global_assignment_prior"]["global_assignment_mode"] == 0.0
    assert out2[0]["module_details"]["global_assignment_prior"]["global_assignment_mode"] == 1.0


def test_constant_module_scores_are_neutral_not_all_ones() -> None:
    out = _normalize_scores({(0, 0): 2.0, (0, 1): 2.0})
    assert out[(0, 0)] == 0.5
    assert out[(0, 1)] == 0.5
