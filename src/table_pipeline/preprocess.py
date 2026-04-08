from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rectify import rectify_document


class PreprocessError(ValueError):
    pass


@dataclass
class PreprocessResult:
    image_bgr: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    diagnostics: dict[str, object]


def preprocess_image(image_path: str) -> PreprocessResult:
    image = cv2.imread(image_path)
    if image is None:
        raise PreprocessError(f"cannot_read_image:{image_path}")

    rect = rectify_document(image)
    gray = rect.gray
    denoise = cv2.fastNlMeansDenoising(gray, h=8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoise)

    bw_inv = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(bw_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return PreprocessResult(
        image_bgr=rect.image_bgr,
        gray=gray,
        enhanced=enhanced,
        binary=cleaned,
        diagnostics={
            "perspective_applied": rect.perspective_applied,
            "deskew_angle": rect.deskew_angle,
        },
    )
