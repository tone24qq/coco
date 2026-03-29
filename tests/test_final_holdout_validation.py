from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "final_holdout_validation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "final_holdout_validation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


REQUIRED_SUMMARY_KEYS = MOD.REQUIRED_SUMMARY_KEYS
leakage_check = MOD.leakage_check
load_snapshot = MOD.load_snapshot
metrics_from_top3 = MOD.metrics_from_top3
snapshot_fingerprint = MOD.snapshot_fingerprint
split_windows = MOD.split_windows
split_final_blocks = MOD.split_final_blocks
evaluate_window = MOD.evaluate_window


def _build_draws(count: int = 80) -> tuple[list[int], list[list[int]]]:
    periods = list(range(1000, 1000 + count))
    numbers: list[list[int]] = []
    for i in range(count):
        base = (i % 60) + 1
        draw = list(range(base, base + 20))
        draw = [((n - 1) % 80) + 1 for n in draw]
        numbers.append(sorted(draw))
    return periods, numbers


def test_mainline_integration_evaluate_window_uses_predict_pipeline() -> None:
    periods, numbers = _build_draws()
    cfg = MOD.AppConfig(min_prediction_draws=10, min_score_threshold=10)
    rows, buckets = evaluate_window(
        numbers, periods, 10, 20, cfg, seed=1, include_baselines=False
    )
    assert rows
    assert "model_top3_at_least_one_hit_rate" in rows[0]
    assert "model" in buckets


def test_split_windows_and_final_blocks_no_leakage() -> None:
    windows = split_windows(
        total_draws=120,
        search_ratio=0.6,
        validation_ratio=0.2,
        final_ratio=0.2,
        min_train_draws=20,
    )
    blocks = split_final_blocks(*windows["final"])
    assert leakage_check(windows, blocks) is True


def test_split_windows_fail_fast_when_insufficient() -> None:
    with pytest.raises(ValueError):
        split_windows(
            total_draws=30,
            search_ratio=0.6,
            validation_ratio=0.2,
            final_ratio=0.2,
            min_train_draws=20,
        )


def test_frozen_snapshot_fingerprint_validation(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    data = {
        "draws": [[100, [1, 2, 3]], [101, [4, 5, 6]]],
        "fingerprint": snapshot_fingerprint(
            [(100, [1, 2, 3]), (101, [4, 5, 6])]
        ),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_snapshot(path)
    assert loaded == [(100, [1, 2, 3]), (101, [4, 5, 6])]


def test_metrics_output_reality_check() -> None:
    top3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    actual = list(range(1, 21))
    metrics = metrics_from_top3(top3, actual)
    expected_keys = {
        "top3_at_least_one_hit_rate",
        "exact_hit@3",
        "exact_hit@10",
        "exact_hit@20",
        "adj_hit_pm1@3",
        "strict_adj_only_pm1@3",
        "mean_min_distance@3",
        "signed_offset_mean@3",
        "overshoot_rate@3",
        "undershoot_rate@3",
    }
    assert expected_keys == set(metrics.keys())
    assert 0.0 <= metrics["top3_at_least_one_hit_rate"] <= 1.0


def test_summary_schema_keys_present() -> None:
    example_summary = {
        "snapshot_source": "frozen_snapshot",
        "snapshot_fingerprint": "abc",
        "search_issue_range": [1, 2],
        "validation_issue_range": [3, 4],
        "final_holdout_issue_range": [5, 8],
        "final_holdout_blocks": [],
        "total_draws_search": 2,
        "total_draws_validation": 2,
        "total_draws_final_holdout": 4,
        "chosen_config": {},
        "final_metrics": {},
        "baseline_metrics": {},
        "block_metrics": [],
        "p_value_vs_frequency": 0.1,
        "bootstrap_ci_vs_frequency": {"low": 0.0, "high": 0.1},
        "leakage_check_passed": True,
        "passed": False,
        "pass_reason": "x",
    }
    assert REQUIRED_SUMMARY_KEYS.issubset(set(example_summary.keys()))


def test_final_holdout_not_part_of_search_validation_ranges() -> None:
    windows = split_windows(
        total_draws=120,
        search_ratio=0.6,
        validation_ratio=0.2,
        final_ratio=0.2,
        min_train_draws=20,
    )
    blocks = split_final_blocks(*windows["final"])
    s0, s1 = windows["search"]
    v0, v1 = windows["validation"]
    f0, f1 = windows["final"]
    assert s1 <= v0
    assert v1 <= f0
    assert blocks[0][0] == f0
    assert blocks[1][1] == f1
