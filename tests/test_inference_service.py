from __future__ import annotations

from src.inference_service import build_cell_candidates, rank_candidates, score_candidates


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
