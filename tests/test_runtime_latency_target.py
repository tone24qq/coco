from __future__ import annotations

import json
import time
from pathlib import Path

from src.inference_service import _run_inference_detailed
from src.scoring_modules import GlobalAssignmentPriorModule


def _build_8x10_masked_board() -> list[list[int]]:
    rows, cols = 8, 10
    cur = 1
    board: list[list[int]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = cur
            cur += 1
            if (r + c) % 2 == 0:
                row.append(-1)
            else:
                row.append(v)
        board.append(row)
    return board


def _build_10x16_masked_board() -> list[list[int]]:
    rows, cols = 10, 16
    cur = 1
    board: list[list[int]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = cur
            cur += 1
            if (r * 2 + c) % 3 == 0:
                row.append(-1)
            else:
                row.append(v)
        board.append(row)
    return board


def _pick_target(board: list[list[int]]) -> int:
    n_total = len(board) * len(board[0])
    opened = {v for row in board for v in row if v != -1}
    for x in range(1, n_total + 1):
        if x not in opened:
            return x
    raise AssertionError("no target")


def _top10_cells(result: dict) -> list[tuple[int, int]]:
    return [(c["row"], c["col"]) for c in result["candidate_cells"][:10]]


def test_fast_mode_latency_under_5_seconds() -> None:
    board = _build_8x10_masked_board()
    target = _pick_target(board)
    rounds = 3
    _run_inference_detailed(board, target, source="latency", apply_reranker_stage=False)
    start = time.perf_counter()
    last = None
    for _ in range(rounds):
        last = _run_inference_detailed(board, target, source="latency", apply_reranker_stage=False)
    elapsed = (time.perf_counter() - start) / rounds
    assert elapsed <= 5.0, f"avg elapsed {elapsed:.3f}s > 5s"
    assert last is not None
    assert last["metadata"]["runtime_mode"] == "fast"


def test_fast_mode_latency_under_5_seconds_10x16() -> None:
    board = _build_10x16_masked_board()
    target = _pick_target(board)
    rounds = 2
    _run_inference_detailed(board, target, source="latency", apply_reranker_stage=False)
    start = time.perf_counter()
    for _ in range(rounds):
        _run_inference_detailed(board, target, source="latency", apply_reranker_stage=False)
    elapsed = (time.perf_counter() - start) / rounds
    assert elapsed <= 5.0, f"avg elapsed {elapsed:.3f}s > 5s"


def test_fast_mode_ranking_reasonable_against_full() -> None:
    board = _build_8x10_masked_board()
    target = _pick_target(board)
    baseline = _run_inference_detailed(
        board,
        target,
        source="full",
        apply_reranker_stage=False,
        module_settings={
            "pairwise_conditional_consistency": {"runtime_mode": "full"},
            "global_assignment_prior": {"assignment_mode": "exact", "top_m_candidates": 8},
        },
    )
    fast = _run_inference_detailed(board, target, source="fast", apply_reranker_stage=False)
    base_top10 = set(_top10_cells(baseline))
    fast_top10 = set(_top10_cells(fast))
    overlap = len(base_top10 & fast_top10) / max(len(base_top10), 1)
    top1_same = (
        baseline["candidate_cells"][0]["row"] == fast["candidate_cells"][0]["row"]
        and baseline["candidate_cells"][0]["col"] == fast["candidate_cells"][0]["col"]
    )
    assert top1_same or overlap >= 0.6


def test_reranker_disabled_does_not_build_feature_rows(monkeypatch) -> None:
    artifact_path = Path("artifacts/reranker_weights.json")
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_text(json.dumps({"enabled": False, "version": "test", "fallback_reason": "disabled_for_test"}))

    def _boom(*args, **kwargs):
        raise AssertionError("feature rows should not be built when reranker disabled")

    monkeypatch.setattr("src.inference_service.build_candidate_feature_rows", _boom)
    board = [[1, -1, 3], [-1, 5, -1]]
    result = _run_inference_detailed(board, 4, source="reranker", apply_reranker_stage=True)
    assert result["metadata"]["ranking_stage"] == "baseline_only"


def test_global_assignment_exact_disabled_when_too_many_candidates() -> None:
    board = _build_8x10_masked_board()
    unopened = [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == -1]
    module = GlobalAssignmentPriorModule(assignment_mode="exact", top_m_candidates=4, exact_max_candidates=20)
    out = module.score(board, unopened, target_number=_pick_target(board))
    anchor_details = [d for d in out.details.values() if d.get("used_compatibility_fallback", 0.0) == 0.0]
    assert anchor_details
    assert all(d.get("used_exact_assignment", 0.0) == 0.0 for d in anchor_details)
    assert any(d.get("exact_forced_off_by_candidate_count", 0.0) > 0.0 for d in anchor_details)
