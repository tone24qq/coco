from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CellClassifyResult:
    label: str
    confidence: float
    features: dict[str, float]
    needs_review: bool


LABELS = ("blank", "printed_number", "scratched_or_occluded", "solid_black")


def classify_cell(cell: np.ndarray) -> CellClassifyResult:
    if cell.size == 0:
        return CellClassifyResult("blank", 0.0, {"ink_ratio": 0.0}, True)

    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(np.mean(bw > 0))
    ncc, _labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA] if ncc > 1 else np.array([])
    comp_count = float(len(comp_areas))
    max_comp = float(np.max(comp_areas)) / float(cell.size) if len(comp_areas) else 0.0

    if ink_ratio < 0.02:
        label = "blank"
        conf = 1.0 - min(1.0, ink_ratio / 0.02)
    elif ink_ratio > 0.75 or max_comp > 0.7:
        label = "solid_black"
        conf = min(1.0, ink_ratio)
    elif comp_count >= 6 or (ink_ratio > 0.30 and max_comp < 0.25):
        label = "scratched_or_occluded"
        conf = 0.55
    else:
        label = "printed_number"
        conf = max(0.4, min(0.95, 1.0 - abs(ink_ratio - 0.16) * 3.5))

    needs_review = conf < 0.65 or label == "scratched_or_occluded"
    return CellClassifyResult(
        label=label,
        confidence=conf,
        features={"ink_ratio": ink_ratio, "component_count": comp_count, "max_component_ratio": max_comp},
        needs_review=needs_review,
    )
