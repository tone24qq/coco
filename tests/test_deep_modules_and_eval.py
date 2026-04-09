from __future__ import annotations

import math
from pathlib import Path

from scripts.run_gogo_formal_eval import _masked_board, load_eval_cases, random_weight_search
from src.inference_service import run_inference
from src.scoring_modules import MODULES


def _board_3x4() -> list[list[int]]:
    return [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]


def test_arbitrary_board_size_infer_and_module_scores_finite() -> None:
    full = _board_3x4()
    masked = [row[:] for row in full]
    masked[1][1] = -1
    masked[2][3] = -1
    res = run_inference(masked, target_number=6, source="unit")
    assert res["status"] == "ok"
    for cand in res["candidate_cells"]:
        assert isinstance(cand["row"], int)
        for m in ("directional_consistency", "line_consistency", "global_assignment_prior"):
            v = cand["module_scores"][m]
            assert 0.0 <= float(v) <= 1.0
            assert math.isfinite(float(v))


def test_masking_ratio_floor_half() -> None:
    full = _board_3x4()
    masked, cells = _masked_board(full, 0.5, __import__("random").Random(1))
    assert len(cells) == 6
    assert sum(1 for row in masked for v in row if v == -1) == 6


def test_weight_tuning_produces_normalized_weights() -> None:
    full = _board_3x4()
    masked = [row[:] for row in full]
    masked[0][1] = -1
    masked[2][2] = -1
    eval_cases = [
        (
            type("Case", (), {"board_id": "x"})(),
            masked,
            2,
            (0, 1),
        )
    ]
    mods = [
        "logic_rule",
        "pattern_model",
        "prior_model",
        "directional_consistency",
        "line_consistency",
        "global_assignment_prior",
    ]
    best, records = random_weight_search(mods, trials=3, seed=7, eval_cases=eval_cases)
    assert records
    assert abs(sum(best.values()) - 1.0) < 1e-6


def test_gogo_missing_fail_fast(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    try:
        load_eval_cases(missing)
        assert False, "expected failure"
    except ValueError as exc:
        assert "does not exist" in str(exc)


def test_global_assignment_safe_fallback_information_insufficient() -> None:
    board = [[-1]]
    res = MODULES["global_assignment_prior"].score(board, [(0, 0)], 1)
    assert (0, 0) in res.scores
    assert 0.0 <= res.scores[(0, 0)] <= 1.0


def test_baseline_only_path_not_broken() -> None:
    board = [[1, -1, 3], [-1, 5, -1]]
    out = run_inference(board, 4, source="unit", apply_reranker_stage=False)
    assert out["metadata"]["ranking_stage"] == "baseline_only"
