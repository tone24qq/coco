from __future__ import annotations

import math

from src.scoring_modules import PairwiseConditionalConsistencyModule


def test_pairwise_bounded_search_not_explosive() -> None:
    board = [[1, -1, -1], [-1, 5, -1], [-1, -1, 9]]
    unopened = [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == -1]
    module = PairwiseConditionalConsistencyModule(
        anchor_top_k_cells=5,
        anchor_top_k_values=5,
        max_pair_trials_per_candidate=7,
    )
    result = module.score(board, unopened, target_number=4)
    for cell in unopened:
        assert result.details[cell]["pair_trials_used"] <= 7


def test_anchor_never_conflicts_with_target_cell_or_value() -> None:
    board = [[1, -1], [-1, 4]]
    unopened = [(0, 1), (1, 0)]
    module = PairwiseConditionalConsistencyModule(max_pair_trials_per_candidate=10)
    result = module.score(board, unopened, target_number=2)
    for cell in unopened:
        best_anchor = (
            int(result.details[cell]["best_anchor_row"]) - 1,
            int(result.details[cell]["best_anchor_col"]) - 1,
        )
        anchor_value = int(result.details[cell]["best_anchor_value"])
        if best_anchor != (-2, -2):
            assert best_anchor != cell
        assert anchor_value != 2


def test_conditional_gain_not_nan_and_non_negative() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    unopened = [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == -1]
    module = PairwiseConditionalConsistencyModule()
    result = module.score(board, unopened, target_number=4)
    for cell in unopened:
        gain = float(result.details[cell]["conditional_gain"])
        assert not math.isnan(gain)
        assert gain >= 0.0
