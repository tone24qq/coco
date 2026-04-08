from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.cell_digit_reader import read_cell_digit


NORMALIZE_MAP = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})


def normalize_text(raw: str) -> str:
    txt = raw.translate(NORMALIZE_MAP)
    txt = txt.replace(",", "").replace(" ", "").strip()
    txt = re.sub(r"[^0-9.-]", "", txt)
    return txt


def ocr_cell(gray: np.ndarray, bbox: tuple[int, int, int, int], max_value: int) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    cell = gray[y0:y1, x0:x1]
    if cell.size == 0:
        return {
            "text": "",
            "confidence": 0.0,
            "is_numeric": False,
            "normalized_value": None,
            "review_needed": True,
            "label": "unknown",
            "top_candidates": [],
        }

    ink_ratio = float(np.mean(cell < 220))
    if ink_ratio < 0.015:
        return {
            "text": "",
            "confidence": 1.0,
            "is_numeric": False,
            "normalized_value": None,
            "review_needed": False,
            "label": "empty",
            "top_candidates": [],
        }

    result = read_cell_digit(cell, max_value=max_value)
    text = str(result.value) if result.value is not None else ""
    norm_txt = normalize_text(text)
    norm_val = int(norm_txt) if norm_txt.isdigit() else None
    is_numeric = norm_val is not None
    label = "number" if is_numeric else "unknown"
    review_needed = (not is_numeric) or float(result.confidence) < 0.62
    return {
        "text": text,
        "confidence": float(result.confidence),
        "is_numeric": bool(is_numeric),
        "normalized_value": norm_val,
        "review_needed": bool(review_needed),
        "label": label,
        "top_candidates": result.top_candidates,
    }
