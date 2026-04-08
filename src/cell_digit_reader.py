from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional runtime dependency
    RapidOCR = None  # type: ignore[assignment]


@dataclass
class DigitReadResult:
    value: int | None
    confidence: float
    top_candidates: list[dict[str, float | int]]
    ocr_backend: str


@lru_cache(maxsize=1)
def _rapid_ocr_engine():
    if RapidOCR is None:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None


def _normalize_cell(img: np.ndarray, size: int = 56) -> np.ndarray:
    if img.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    den = cv2.fastNlMeansDenoising(img, h=12)
    _, bw = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    x0, x1 = max(0, int(xs.min()) - 2), min(bw.shape[1], int(xs.max()) + 3)
    y0, y1 = max(0, int(ys.min()) - 2), min(bw.shape[0], int(ys.max()) + 3)
    crop = bw[y0:y1, x0:x1]
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = min((size - 4) / max(crop.shape[0], 1), (size - 4) / max(crop.shape[1], 1))
    nh, nw = max(1, int(crop.shape[0] * scale)), max(1, int(crop.shape[1] * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    yoff, xoff = (size - nh) // 2, (size - nw) // 2
    canvas[yoff : yoff + nh, xoff : xoff + nw] = resized
    return canvas


@lru_cache(maxsize=16)
def _fallback_templates(max_value: int, size: int = 56) -> dict[int, np.ndarray]:
    tmps: dict[int, np.ndarray] = {}
    for v in range(1, max_value + 1):
        c = np.zeros((size, size), dtype=np.uint8)
        txt = str(v)
        fs = 0.75 if v < 100 else 0.55
        ts, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        x = max(1, (size - ts[0]) // 2)
        y = max(ts[1] + 1, (size + ts[1]) // 2)
        cv2.putText(c, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, 2, cv2.LINE_AA)
        tmps[v] = c
    return tmps


def read_cell_digit(cell: np.ndarray, max_value: int) -> DigitReadResult:
    obs = _normalize_cell(cell)
    if np.sum(obs) < 20:
        return DigitReadResult(None, 0.0, [], "fallback_template")

    engine = _rapid_ocr_engine() if max_value <= 40 else None
    if engine is not None:
        try:
            ocr_results, _ = engine(obs)
        except Exception:
            ocr_results = None
        if ocr_results:
            digit_candidates = []
            for item in ocr_results:
                if len(item) < 3:
                    continue
                text = str(item[1]).strip()
                score = float(item[2])
                if text.isdigit():
                    value = int(text)
                    if 1 <= value <= max_value:
                        digit_candidates.append({"value": value, "score": score})
            if digit_candidates:
                digit_candidates.sort(key=lambda x: float(x["score"]), reverse=True)
                top = digit_candidates[:5]
                best = top[0]
                conf = max(0.0, min(1.0, float(best["score"])))
                value = int(best["value"]) if conf >= 0.40 else None
                return DigitReadResult(
                    value=value,
                    confidence=conf,
                    top_candidates=top,
                    ocr_backend="rapidocr",
                )

    templates = _fallback_templates(max_value)
    scored: list[tuple[int, float]] = []
    for v, tmp in templates.items():
        score = float(cv2.matchTemplate(obs, tmp, cv2.TM_CCOEFF_NORMED)[0][0])
        scored.append((v, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:5]
    best_v, best_s = top[0]
    second_s = top[1][1] if len(top) > 1 else -1.0
    margin = best_s - second_s
    conf = max(0.0, min(1.0, ((best_s + 1.0) / 2.0) * 0.75 + max(0.0, margin) * 0.25))
    top_candidates = [{"value": int(v), "score": float(s)} for v, s in top]
    value = int(best_v) if conf >= 0.50 else None
    return DigitReadResult(value=value, confidence=conf, top_candidates=top_candidates, ocr_backend="fallback_template")
