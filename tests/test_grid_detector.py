import numpy as np

from src.grid_detector import detect_grid


def test_grid_detector_fail_fast_on_blank() -> None:
    img = np.full((200, 300), 255, dtype=np.uint8)
    try:
        detect_grid(img)
    except ValueError as exc:
        assert "board" in str(exc) or "grid" in str(exc)
    else:
        raise AssertionError("expected ValueError")
