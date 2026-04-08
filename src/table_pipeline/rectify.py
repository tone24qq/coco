from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RectifyResult:
    image_bgr: np.ndarray
    gray: np.ndarray
    perspective_applied: bool
    deskew_angle: float


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _find_main_quad(gray: np.ndarray) -> np.ndarray | None:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 40, 150)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    x, y, w, h = cv2.boundingRect(best)
    if w < 40 or h < 40:
        return None
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    ys, xs = np.where(bw > 0)
    if len(xs) < 200:
        return gray, 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = float(rect[-1])
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.4:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def rectify_document(image_bgr: np.ndarray) -> RectifyResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = _find_main_quad(gray)
    perspective_applied = False
    if quad is not None:
        ordered = _order_quad(quad)
        width = int(max(np.linalg.norm(ordered[0] - ordered[1]), np.linalg.norm(ordered[2] - ordered[3])))
        height = int(max(np.linalg.norm(ordered[0] - ordered[3]), np.linalg.norm(ordered[1] - ordered[2])))
        if width > 100 and height > 100:
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            m = cv2.getPerspectiveTransform(ordered, dst)
            image_bgr = cv2.warpPerspective(image_bgr, m, (width, height))
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            perspective_applied = True
    gray, angle = _deskew(gray)
    if abs(angle) > 0.01:
        h, w = image_bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image_bgr = cv2.warpAffine(
            image_bgr,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return RectifyResult(
        image_bgr=image_bgr,
        gray=gray,
        perspective_applied=perspective_applied,
        deskew_angle=float(angle),
    )
