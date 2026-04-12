from __future__ import annotations

import pytest

from src.inference_service import InferenceError, _run_inference_detailed
from src.scoring_modules import (
    GlobalAssignmentPriorModule,
    PriorModelModule,
    _cell_number_compatibility,
    _directional_components,
    _line_components,
    build_modules,
)


def _committee_cfg() -> dict:
    return {
        "type": "committee_weighted_sum",
        "weighting_mode": "equal_informative",
        "allow_abstain": True,
        "preserve_diagnostics": True,
        "judge": {"enabled": False, "use_for_primary_ranking": False},
        "diagnostics": {
            "include_vote_features": True,
            "include_rank_features": True,
            "include_score_features": True,
        },
    }


def test_prior_model_disabled_in_committee() -> None:
    with pytest.raises(InferenceError):
        _run_inference_detailed(
            [[1, -1, 3], [-1, 5, -1]],
            4,
            source="t",
            apply_reranker_stage=False,
            module_weights={"prior_model": 1.0},
            aggregator_config=_committee_cfg(),
        )


def test_global_assignment_no_prior_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("prior model should not be used by global assignment")

    monkeypatch.setattr(PriorModelModule, "score", _boom)
    module = GlobalAssignmentPriorModule(assignment_mode="greedy", top_m_candidates=2)
    res = module.score([[1, -1, 3], [-1, 5, -1]], [(0, 1), (1, 0), (1, 1)], 4)
    assert len(res.scores) == 3


def test_cell_number_compatibility_without_prior() -> None:
    board = [[1, -1, 3], [-1, 5, -1], [7, -1, 9]]
    cell = (1, 0)
    number = 4
    directional = _directional_components(board, cell, number)["directional_consistency"]
    line = _line_components(board, cell, number)["line_consistency"]
    compat = _cell_number_compatibility(board, cell, number)
    assert abs(compat - (0.5 * directional + 0.5 * line)) < 1e-9


def test_local_arithmetic_relation_uses_target_near_seeds() -> None:
    module = build_modules({"local_arithmetic_relation": {}})["local_arithmetic_relation"]
    board = [[89, -1, 91], [30, -1, 40], [70, -1, 110]]
    cells = [(0, 1), (1, 1), (2, 1)]
    res = module.score(board, cells, 90)
    near_seed_top = max(cells, key=lambda c: res.details[c]["target_near_seed_strength"])
    assert near_seed_top == (0, 1)
    assert res.details[(0, 1)]["target_near_seed_strength"] >= res.details[(2, 1)]["target_near_seed_strength"]


def test_local_arithmetic_relation_is_candidate_sensitive() -> None:
    module = build_modules({"local_arithmetic_relation": {}})["local_arithmetic_relation"]
    board = [[10, -1, 20], [35, -1, 50], [70, -1, 90]]
    cells = [(0, 1), (1, 1), (2, 1)]
    res = module.score(board, cells, 30)
    scores = {res.scores[c] for c in cells}
    assert len(scores) > 1


def test_local_arithmetic_relation_row_col_gap_improvement_changes_by_cell() -> None:
    module = build_modules({"local_arithmetic_relation": {}})["local_arithmetic_relation"]
    board = [[12, -1, 18], [30, -1, 60], [33, -1, 36]]
    cells = [(0, 1), (1, 1), (2, 1)]
    res = module.score(board, cells, 24)
    row_improvements = {res.details[c]["row_gap_improvement"] for c in cells}
    col_improvements = {res.details[c]["col_gap_improvement"] for c in cells}
    assert len(row_improvements) > 1 or len(col_improvements) > 1


def test_local_arithmetic_relation_abstains_when_no_local_signal() -> None:
    module = build_modules({"local_arithmetic_relation": {}})["local_arithmetic_relation"]
    board = [[1, -1], [-1, -1]]
    cells = [(0, 1), (1, 0), (1, 1)]
    res = module.score(board, cells, 4)
    assert all(res.informative_cells[c] == 0.0 for c in cells)
    assert all(res.scores[c] == 0.5 for c in cells)


def test_sparse_mode_uses_only_informative_modules() -> None:
    board = [[1, -1], [-1, -1]]
    out = _run_inference_detailed(
        board,
        4,
        source="t",
        apply_reranker_stage=False,
        module_weights={"logic_rule": 0.5, "local_arithmetic_relation": 0.5},
        aggregator_config=_committee_cfg(),
    )
    cell = out["candidate_cells"][0]
    assert cell["module_informative"]["local_arithmetic_relation"] == 0.0
    assert "local_arithmetic_relation" not in cell["module_effective_weights"]


def test_no_position_bias_when_all_modules_abstain() -> None:
    out = _run_inference_detailed(
        [[1, -1], [-1, -1]],
        4,
        source="t",
        apply_reranker_stage=False,
        module_weights={"local_arithmetic_relation": 1.0},
        aggregator_config=_committee_cfg(),
    )
    scores = {c["score"] for c in out["candidate_cells"]}
    assert scores == {0.5}
    assert out["metadata"]["no_informative_modules"] is True
