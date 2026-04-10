from __future__ import annotations

import numpy as np

from src.masking_eval.backtest import _metrics_from_ranks, generate_masked, run_backtest
from src.masking_eval.discovery import run_module_discovery


class DummyBoard:
    def __init__(self, board_id: str, size_class: str, order_index: int) -> None:
        shape_map = {"20": (4, 5), "120": (10, 12), "160": (10, 16)}
        rows, cols = shape_map.get(size_class, (10, 8))
        self.board_id = board_id
        self.size_class = size_class
        self.order_index = order_index
        self.grid = np.arange(1, rows * cols + 1).reshape(rows, cols)


def test_synthetic_masking_is_50pct_and_reproducible() -> None:
    grid = np.arange(1, 161).reshape(10, 16)
    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    masked_a, targets_a = generate_masked(grid, rng_a)
    masked_b, targets_b = generate_masked(grid, rng_b)
    assert len(targets_a) == grid.size // 2
    assert targets_a == targets_b
    assert np.array_equal(masked_a, masked_b)


def test_top10_metrics_are_correct() -> None:
    ranks = [1, 2, 2, 10, 11]
    cands = [80, 80, 80, 80, 80]
    m = _metrics_from_ranks(ranks, cands)
    assert m["cumulative_top1_hit_rate"] == 0.2
    assert m["cumulative_top2_hit_rate"] == 0.6
    assert m["cumulative_top10_hit_rate"] == 0.8
    assert m["exact_rank2_hit_rate"] == 0.4
    assert m["overall_top10_hit_rate"] == 0.8


def test_weight_search_outputs_best_weights_and_trials() -> None:
    boards = [DummyBoard(f"b{i}", "20", i) for i in range(6)]
    result = run_backtest(boards, folds=3, repeats=2, seed=2026, modules=["focus", "tail"], n_trials=5)
    assert "best_weights" in result
    assert result["best_weights"]
    assert len(result["trial_leaderboard"]) > 0


def test_module_rejected_without_improvement() -> None:
    boards = [DummyBoard(f"b{i}", "20", i) for i in range(6)]
    result = run_module_discovery(
        boards=boards,
        folds=3,
        repeats=2,
        seed=2026,
        n_trials=2,
        candidate_modules=["local_arith_completion"],
    )
    assert result["leaderboard"][0]["keep"] is False
    assert result["dropped_modules"] == ["local_arith_completion"]


def test_legacy_sizes_not_broken() -> None:
    boards = [
        DummyBoard("a", "120", 0),
        DummyBoard("b", "120", 1),
        DummyBoard("c", "120", 2),
        DummyBoard("d", "120", 3),
    ]
    result = run_backtest(boards, folds=2, repeats=2, seed=2026, modules=["focus", "tail"], n_trials=3)
    assert result["anti_leakage_checks"] == "passed"
    assert result["num_boards"] == 4
