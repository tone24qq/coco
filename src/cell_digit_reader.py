from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
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


def _normalize_cell(img: np.ndarray, size: int = 64) -> np.ndarray:
    if img.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    den = cv2.fastNlMeansDenoising(img, h=9)
    _, bw = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    x0, x1 = max(0, int(xs.min()) - 2), min(bw.shape[1], int(xs.max()) + 3)
    y0, y1 = max(0, int(ys.min()) - 2), min(bw.shape[0], int(ys.max()) + 3)
    crop = bw[y0:y1, x0:x1]
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = min((size - 6) / max(crop.shape[0], 1), (size - 6) / max(crop.shape[1], 1))
    nh, nw = max(1, int(crop.shape[0] * scale)), max(1, int(crop.shape[1] * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    yoff, xoff = (size - nh) // 2, (size - nw) // 2
    canvas[yoff : yoff + nh, xoff : xoff + nw] = resized
    return canvas


@lru_cache(maxsize=16)
def _fallback_templates(max_value: int, size: int = 64) -> dict[int, np.ndarray]:
    tmps: dict[int, np.ndarray] = {}
    for v in range(1, max_value + 1):
        c = np.zeros((size, size), dtype=np.uint8)
        txt = str(v)
        fs = 0.72 if v < 100 else 0.52
        ts, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        x = max(1, (size - ts[0]) // 2)
        y = max(ts[1] + 1, (size + ts[1]) // 2)
        cv2.putText(c, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, 2, cv2.LINE_AA)
        tmps[v] = c
    return tmps


def read_cell_digit(cell: np.ndarray, max_value: int) -> DigitReadResult:
    obs = _normalize_cell(cell)
    if np.sum(obs) < 20:
        return DigitReadResult(None, 0.0, [], "rapidocr_unavailable")

    cands: dict[int, float] = {}
    backend = "rapidocr_template_rerank"
    engine = _rapid_ocr_engine()
    used_rapid = False
    if engine is not None:
        try:
            ocr_results, _ = engine(obs)
        except Exception:
            ocr_results = None
        if ocr_results:
            for item in ocr_results:
                if len(item) < 3:
                    continue
                text = str(item[1]).strip()
                score = float(item[2])
                if text.isdigit():
                    v = int(text)
                    if 1 <= v <= max_value:
                        cands[v] = max(cands.get(v, 0.0), score)
                        used_rapid = True

    templates = _fallback_templates(max_value)
    scored: list[tuple[int, float]] = []
    for v, tmp in templates.items():
        score = float(cv2.matchTemplate(obs, tmp, cv2.TM_CCOEFF_NORMED)[0][0])
        scored.append((v, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    for rank, (v, s) in enumerate(scored[:8]):
        tscore = max(0.0, min(1.0, (s + 1.0) / 2.0))
        blend = 0.35 * tscore
        if v in cands:
            blend += 0.65 * cands[v]
        cands[v] = max(cands.get(v, 0.0), blend * (1.0 - rank * 0.04))

    if used_rapid:
        backend = "rapidocr"
    if not cands:
        return DigitReadResult(None, 0.0, [], backend)
    ordered = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:5]
    top_candidates = [{"value": int(v), "score": float(s)} for v, s in ordered]
    best_v, best_s = ordered[0]
    value = int(best_v) if best_s >= 0.34 else None
    return DigitReadResult(value=value, confidence=float(best_s), top_candidates=top_candidates, ocr_backend=backend)
