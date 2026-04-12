from __future__ import annotations

from src.vector_modules import (
    compute_local_tail_evidence,
    connectivity_heatmap_vectorized,
    difference_trend_vectorized,
    focus_score_vectorized,
    mirror_sequences_vectorized,
    skip_patterns_vectorized,
    tail_analyzer_vectorized,
)


def _sample_board() -> tuple[list[list[int]], list[tuple[int, int]]]:
    board = [[1, -1, 3, -1], [-1, 6, -1, 8], [9, -1, 11, -1], [-1, 14, -1, 16]]
    unopened = [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == -1]
    return board, unopened


def _assert_scores(scores: dict, unopened: list[tuple[int, int]]) -> None:
    assert set(scores.keys()) == set(unopened)
    assert all(0.0 <= float(v) <= 1.0 for v in scores.values())


def test_vector_modules_score_range_and_coverage() -> None:
    board, unopened = _sample_board()
    target = 7
    _assert_scores(focus_score_vectorized(board, unopened, window_size=3), unopened)
    _assert_scores(connectivity_heatmap_vectorized(board, unopened, decay="inverse_distance"), unopened)
    _assert_scores(difference_trend_vectorized(board, unopened, target), unopened)
    _assert_scores(skip_patterns_vectorized(board, unopened, target), unopened)
    _assert_scores(mirror_sequences_vectorized(board, unopened, target), unopened)
    _assert_scores(tail_analyzer_vectorized(board, unopened, target, window_size=3), unopened)


def test_tail_evidence_gate_and_tail_analyzer_cap() -> None:
    board = [[21, 31, -1], [2, -1, 41], [-1, 11, -1]]
    unopened = [(0, 2), (1, 1), (2, 0), (2, 2)]
    target = 1
    scores = tail_analyzer_vectorized(board, unopened, target, window_size=3)
    evidence_center = compute_local_tail_evidence(board, (1, 1), target, window_size=3)
    assert evidence_center["strong_tail_signal"] == 1.0
    assert 0.5 <= scores[(1, 1)] <= 0.72
