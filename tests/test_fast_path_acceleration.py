from __future__ import annotations

from src.scoring_modules import (
    DirectionalConsistencyModule,
    GlobalAssignmentPriorModule,
    LineConsistencyModule,
    LogicRuleModule,
)


def test_fast_path_close_to_fallback_ranking() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    unopened = [(0, 1), (1, 0), (1, 2)]
    fast = LogicRuleModule(fast_enabled=True).score(board, unopened, 4).scores
    slow = LogicRuleModule(fast_enabled=False).score(board, unopened, 4).scores
    fast_rank = sorted(unopened, key=lambda c: fast[c], reverse=True)
    slow_rank = sorted(unopened, key=lambda c: slow[c], reverse=True)
    assert fast_rank[0] == slow_rank[0]


def test_assignment_reduced_candidate_set_flagged() -> None:
    board = [[1, -1, -1], [-1, 5, -1], [-1, -1, 9]]
    unopened = [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == -1]
    result = GlobalAssignmentPriorModule(assignment_mode="greedy", top_m_candidates=2).score(board, unopened, 4)
    skipped = [d for d in result.details.values() if d.get("reduced_assignment_skipped", 0.0) > 0.0]
    assert skipped
    score_values = [result.scores[cell] for cell in unopened]
    assert len(set(round(v, 6) for v in score_values)) > 1


def test_numba_unavailable_fallback_does_not_crash() -> None:
    board = [[1, -1], [-1, 4]]
    unopened = [(0, 1), (1, 0)]
    out = LogicRuleModule(fast_enabled=True).score(board, unopened, 2)
    assert len(out.scores) == 2


def test_directional_fast_vs_slow_top1_consistent() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    unopened = [(0, 1), (1, 0), (1, 2)]
    fast = DirectionalConsistencyModule(fast_enabled=True).score(board, unopened, 4).scores
    slow = DirectionalConsistencyModule(fast_enabled=False).score(board, unopened, 4).scores
    assert max(unopened, key=lambda c: fast[c]) == max(unopened, key=lambda c: slow[c])


def test_line_fast_vs_slow_top1_consistent() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    unopened = [(0, 1), (1, 0), (1, 2)]
    fast = LineConsistencyModule(fast_enabled=True).score(board, unopened, 4).scores
    slow = LineConsistencyModule(fast_enabled=False).score(board, unopened, 4).scores
    assert max(unopened, key=lambda c: fast[c]) == max(unopened, key=lambda c: slow[c])
