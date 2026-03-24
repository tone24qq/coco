import pandas as pd
import pytest

from src.build_rank_windows import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    build_inference_window,
)


def _make_history(start: int = 1000, end: int = 1120) -> pd.DataFrame:
    rows = []
    for issue in range(start, end):
        rows.append(
            {
                "issue": str(issue),
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    return pd.DataFrame(rows)


def test_build_rank_windows_80_candidates() -> None:
    history = _make_history()
    window = build_inference_window(history, window_size=100)
    if window.features.shape != (80, len(FEATURE_NAMES)):
        pytest.fail("feature shape mismatch")
    if window.feature_version != FEATURE_VERSION:
        pytest.fail("feature version mismatch")


def test_build_rank_windows_no_future_data_fail() -> None:
    history = _make_history(1000, 1001)
    with pytest.raises(ValueError, match="at least 2"):
        build_inference_window(history, window_size=100)
