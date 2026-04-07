from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class DigitReadResult:
    value: int | None
    confidence: float


def _prep(img: np.ndarray, size: int = 40) -> np.ndarray:
    if img.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    x0, x1 = max(0, int(xs.min()) - 1), min(bw.shape[1], int(xs.max()) + 2)
    y0, y1 = max(0, int(ys.min()) - 1), min(bw.shape[0], int(ys.max()) + 2)
    crop = bw[y0:y1, x0:x1]
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = min((size - 4) / max(crop.shape[0], 1), (size - 4) / max(crop.shape[1], 1))
    nh = max(1, int(crop.shape[0] * scale))
    nw = max(1, int(crop.shape[1] * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    yoff = (size - nh) // 2
    xoff = (size - nw) // 2
    canvas[yoff : yoff + nh, xoff : xoff + nw] = resized
    return canvas


@lru_cache(maxsize=16)
def _templates(max_value: int, size: int = 40) -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for val in range(1, max_value + 1):
        canvas = np.zeros((size, size), dtype=np.uint8)
        txt = str(val)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6 if val < 100 else 0.45
        thick = 1
        ts, _ = cv2.getTextSize(txt, font, fs, thick)
        x = max(1, (size - ts[0]) // 2)
        y = max(ts[1] + 1, (size + ts[1]) // 2)
        cv2.putText(canvas, txt, (x, y), font, fs, 255, thick, cv2.LINE_AA)
        templates[val] = canvas
    return templates


def read_cell_digit(cell: np.ndarray, max_value: int) -> DigitReadResult:
    obs = _prep(cell)
    if np.sum(obs) < 20:
        return DigitReadResult(value=None, confidence=0.0)
    tpls = _templates(max_value)
    best: Tuple[int | None, float] = (None, -1.0)
    second = -1.0
    for val, tpl in tpls.items():
        score = float(cv2.matchTemplate(obs, tpl, cv2.TM_CCOEFF_NORMED)[0][0])
        if score > best[1]:
            second = best[1]
            best = (val, score)
        elif score > second:
            second = score
    margin = best[1] - max(second, -1.0)
    conf = max(0.0, min(1.0, (best[1] + 1.0) / 2.0 * 0.7 + max(margin, 0.0) * 0.3))
    if conf < 0.55:
        return DigitReadResult(value=None, confidence=conf)
    return DigitReadResult(value=best[0], confidence=conf)
