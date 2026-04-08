from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class PreprocessError(ValueError):
    pass


@dataclass
class PreprocessResult:
    image_bgr: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray


def preprocess_image(image_path: str) -> PreprocessResult:
    image = cv2.imread(image_path)
    if image is None:
        raise PreprocessError(f"cannot_read_image:{image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoise = cv2.fastNlMeansDenoising(gray, h=9)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoise)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    return PreprocessResult(image_bgr=image, gray=gray, enhanced=enhanced, binary=binary)
