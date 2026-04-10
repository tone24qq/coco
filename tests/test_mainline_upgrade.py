from __future__ import annotations

import json
import math
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

from src.board_geometry import anti_diagonal_cells, main_diagonal_cells
from src.inference_service import run_inference
from src.mainline_eval import (
    CORE_MODULES,
    FullBoardRecord,
    discover_full_boards,
    discover_full_boards_with_audit,
    mask_full_board,
    random_weight_candidates,
    run_weighted_eval,
)
from src.scoring_modules import MODULES, _line_components


def test_any_size_board_parse_infer_evaluate() -> None:
    board = [[1, 2, 3], [4, -1, 6], [7, 8, -1], [10, 11, 12]]
    result = run_inference(board, target_number=5, source="test")
    assert result["status"] == "ok"
    assert result["board_shape"] == {"rows": 4, "cols": 3}
    assert len(result["candidate_cells"]) == 2


def test_masking_ratio_50pct_floor() -> None:
    full = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    masked, masked_cells = mask_full_board(full, masking_ratio=0.5, seed=123)
    assert len(masked_cells) == math.floor(9 * 0.5)
    assert sum(1 for row in masked for v in row if v == -1) == 4


def test_new_modules_scores_are_finite() -> None:
    board = [[1, -1, 3], [-1, 5, -1], [7, 8, 9]]
    unopened = [(0, 1), (1, 0), (1, 2)]
    target = 4
    for module in ("directional_consistency", "line_consistency", "global_assignment_prior"):
        result = MODULES[module].score(board, unopened, target)
        for value in result.scores.values():
            assert 0.0 <= value <= 1.0
            assert math.isfinite(value)


def test_weight_search_generates_normalized_weights() -> None:
    candidates = random_weight_candidates(CORE_MODULES, trials=5, seed=7)
    assert candidates
    for w in candidates:
        assert abs(sum(w.values()) - 1.0) < 1e-9


def test_missing_input_dir_fail_fast(tmp_path: Path) -> None:
    missing = tmp_path / "no_gogo_here"
    try:
        discover_full_boards(missing)
    except FileNotFoundError as exc:
        assert "input-dir not found" in str(exc)
    else:
        raise AssertionError("expected fail-fast FileNotFoundError")


def test_global_assignment_prior_safe_fallback() -> None:
    board = [[-1]]
    result = MODULES["global_assignment_prior"].score(board, [(0, 0)], 1)
    assert result.scores[(0, 0)] == 0.5


def test_baseline_only_path_not_broken() -> None:
    result = run_inference([[1, -1, 3], [-1, 5, -1]], 4, source="t", apply_reranker_stage=False)
    assert result["metadata"]["ranking_stage"] == "baseline_only"


def test_gogo_eval_script_outputs_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "gogo"
    input_dir.mkdir()
    (input_dir / "boards.json").write_text(
        json.dumps([{"board_id": "b1", "board": [[1, 2], [3, 4]]}]),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_gogo_mainline_eval.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--repeats",
            "1",
            "--weight-trials",
            "2",
        ]
    )
    assert (out_dir / "per_case_results.csv").exists()
    assert (out_dir / "summary.json").exists()


def test_discover_full_boards_supports_json_csv_txt(tmp_path: Path) -> None:
    root = tmp_path / "gogo"
    root.mkdir()
    (root / "a.json").write_text(json.dumps([{"board_id": "j1", "board": [[1, 2], [3, 4]]}]), encoding="utf-8")
    (root / "b.csv").write_text('board_id,board\nc1,"[[1,2],[3,4]]"\n', encoding="utf-8")
    (root / "c.txt").write_text("board_id:t1\n1 2\n3 4\n", encoding="utf-8")
    boards = discover_full_boards(root)
    ids = {x.board_id for x in boards}
    assert {"j1", "c1", "t1"}.issubset(ids)


def test_weight_search_eval_defaults_to_reranker_disabled() -> None:
    calls = []

    def _fake_run_inference(*args, **kwargs):
        calls.append(kwargs.get("apply_reranker_stage"))
        return {
            "candidate_cells": [
                {"row": 1, "col": 1, "score": 1.0, "confidence_1_to_100": 99.0},
            ]
        }

    with patch("src.mainline_eval.run_inference", side_effect=_fake_run_inference):
        run_weighted_eval(
            boards=[FullBoardRecord(board_id="b", board=[[1]], source="s")],
            weights={m: 1.0 / len(CORE_MODULES) for m in CORE_MODULES},
            masking_ratio=1.0,
            repeats=1,
            seed=1,
        )
    assert calls and all(flag is False for flag in calls)


def test_global_assignment_exact_path() -> None:
    board = [[1, -1], [-1, 4]]
    result = MODULES["global_assignment_prior"].score(board, [(0, 1), (1, 0)], 2)
    cell_details = result.details[(0, 1)]
    assert cell_details["used_exact_assignment"] in (0.0, 1.0)
    assert cell_details["global_assignment_mode"] == 1.0


def test_global_assignment_greedy_fallback_path() -> None:
    module = MODULES["global_assignment_prior"].__class__(assignment_mode="exact")
    with patch("src.scoring_modules.linear_sum_assignment", None):
        result = module.score([[1, -1], [-1, 4]], [(0, 1), (1, 0)], 2)
    details = result.details[(0, 1)]
    assert details["used_greedy_fallback"] == 1.0


def test_discover_full_boards_skips_bad_json_csv_txt_without_crashing(tmp_path: Path) -> None:
    root = tmp_path / "gogo"
    root.mkdir()
    (root / "ok.json").write_text(json.dumps([{"board_id": "ok", "board": [[1, 2], [3, 4]]}]), encoding="utf-8")
    (root / "bad.json").write_text("{oops", encoding="utf-8")
    (root / "bad.csv").write_text("board_id,board\nx,not_json\n", encoding="utf-8")
    (root / "bad.txt").write_text("board_id:t\n1 a\n", encoding="utf-8")
    artifacts = discover_full_boards_with_audit(root)
    assert any(b.board_id == "ok" for b in artifacts.boards)
    assert artifacts.invalid_reasons


def test_shared_diagonal_helper_consistent_between_features_and_scoring() -> None:
    board = [[1, -1, 3], [4, -1, 6], [7, 8, 9]]
    component = _line_components(board, (1, 1), 5)
    main_diag = main_diagonal_cells(3, 3)
    anti_diag = anti_diagonal_cells(3, 3)
    assert (1, 1) in main_diag
    assert (1, 1) in anti_diag
    assert component["main_diag_score"] >= 0.0


def test_summary_contains_apply_reranker_stage_and_settings_snapshot(tmp_path: Path) -> None:
    input_dir = tmp_path / "gogo2"
    input_dir.mkdir()
    (input_dir / "boards.json").write_text(
        json.dumps([{"board_id": "b1", "board": [[1, 2], [3, 4]]}]),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out2"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_gogo_mainline_eval.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--weight-trials",
            "1",
        ]
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "apply_reranker_stage" in summary
    assert "module_settings_snapshot" in summary
