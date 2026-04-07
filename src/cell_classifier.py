from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CellClassifyResult:
    is_filled: bool
    confidence: float
    ink_ratio: float


def classify_cell_empty_or_filled(cell: np.ndarray) -> CellClassifyResult:
    if cell.size == 0:
        return CellClassifyResult(is_filled=False, confidence=0.0, ink_ratio=0.0)
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(np.mean(bw > 0))
    # conservative threshold: mostly blank stays empty
    is_filled = ink_ratio > 0.035
    margin = abs(ink_ratio - 0.035)
    conf = max(0.0, min(1.0, margin / 0.10))
    return CellClassifyResult(is_filled=is_filled, confidence=conf, ink_ratio=ink_ratio)
