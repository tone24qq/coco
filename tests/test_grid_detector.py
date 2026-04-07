import numpy as np
import cv2

from src.grid_detector import GridDetectionError, detect_grid
from src.ticket_specs import get_ticket_spec


def test_grid_detector_fail_fast_on_blank() -> None:
    img = np.full((200, 300), 255, dtype=np.uint8)
    try:
        detect_grid(img, get_ticket_spec("20"))
    except GridDetectionError as exc:
        assert "perspective" in str(exc) or "shape" in str(exc)
    else:
        raise AssertionError("expected GridDetectionError")


def test_grid_detector_80_shape_matches_spec() -> None:
    gray = cv2.imread("gogo/80/NL230505019.jpg", cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    det = detect_grid(gray, get_ticket_spec("80"))
    assert (det.row_count, det.col_count) == (10, 8)
