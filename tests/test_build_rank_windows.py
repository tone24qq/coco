import pandas as pd
import pytest

from src.build_rank_windows import build_inference_window


def test_build_rank_windows_80_candidates() -> None:
    rows = []
    for issue in range(1000, 1105):
        rows.append(
            {
                "issue": str(issue),
                "draw_time": "2026-01-01",
                **{f"n{i}": ((issue + i) % 80) + 1 for i in range(1, 21)},
            }
        )
    history = pd.DataFrame(rows)
    window = build_inference_window(history, window_size=100)
    if window.features.shape != (80, 7):
        pytest.fail("window feature shape mismatch")


def test_build_rank_windows_no_future_data_fail() -> None:
    history = pd.DataFrame(
        [
            {
                "issue": "1001",
                "draw_time": "2026-01-01",
                **{f"n{i}": i for i in range(1, 21)},
            }
        ]
    )
    with pytest.raises(ValueError, match="at least 2"):
        build_inference_window(history, window_size=100)
