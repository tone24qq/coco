import numpy as np
import cv2

import src.cell_digit_reader as cell_digit_reader


def test_read_cell_digit_rapidocr_priority(monkeypatch) -> None:
    class FakeOCR:
        def __call__(self, _img):
            return ([[None, "17", 0.91]], 0.01)

    monkeypatch.setattr(cell_digit_reader, "_rapid_ocr_engine", lambda: FakeOCR())
    img = np.full((64, 64), 255, dtype=np.uint8)
    cv2.putText(img, "17", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    result = cell_digit_reader.read_cell_digit(img, max_value=20)
    assert result.ocr_backend == "rapidocr"
    assert result.value == 17
    assert result.confidence >= 0.9
