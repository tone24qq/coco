from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .ticket_specs import TicketSpec


@dataclass
class GridDetection:
    board_image: np.ndarray
    board_bbox: Tuple[int, int, int, int]
    row_count: int
    col_count: int
    row_lines: list[int]
    col_lines: list[int]
    board_confidence: float
    warp_confidence: float
    shape_confidence: float
    cell_boxes: list[dict[str, int]]


class GridDetectionError(ValueError):
    pass


def _find_board_quad(
    gray: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise GridDetectionError("perspective_unstable")

    h, w = gray.shape
    img_area = float(h * w)
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.02 * peri, True)
    x, y, bw, bh = cv2.boundingRect(best)
    area_ratio = float((bw * bh) / max(img_area, 1.0))
    if area_ratio < 0.08:
        _, bw2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(bw2 > 0)
        if len(xs) < 50 or len(ys) < 50:
            raise GridDetectionError("perspective_unstable")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        approx = np.array(
            [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32
        )
        x, y, bw, bh = x0, y0, x1 - x0, y1 - y0
    if len(approx) != 4:
        hull = cv2.convexHull(best)
        peri2 = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri2, True)
    if len(approx) != 4:
        raise GridDetectionError("perspective_unstable")
    pts = approx.reshape(4, 2).astype(np.float32)
    return pts, (x, y, bw, bh), min(1.0, area_ratio + 0.1)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _warp_to_expected(
    gray: np.ndarray, quad: np.ndarray, expected_rows: int, expected_cols: int
) -> tuple[np.ndarray, float]:
    quad = _order_quad(quad)
    cell = 64
    out_w = expected_cols * cell
    out_h = expected_rows * cell
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(gray, m, (out_w, out_h))
    det = float(abs(np.linalg.det(m[:2, :2])))
    conf = max(0.0, min(1.0, 1.0 / (1.0 + abs(np.log(max(det, 1e-6))))))
    return warped, conf


def detect_grid(gray: np.ndarray, spec: TicketSpec) -> GridDetection:
    quad, bbox, board_conf = _find_board_quad(gray)
    warped, warp_conf = _warp_to_expected(
        gray, quad, spec.expected_rows, spec.expected_cols
    )

    row_lines = list(np.linspace(0, warped.shape[0], spec.expected_rows + 1, dtype=int))
    col_lines = list(np.linspace(0, warped.shape[1], spec.expected_cols + 1, dtype=int))
    row_count = len(row_lines) - 1
    col_count = len(col_lines) - 1
    if (row_count, col_count) != spec.expected_shape:
        raise GridDetectionError("shape_mismatch")

    edges = cv2.Canny(warped, 60, 180)
    edge_density = float(np.mean(edges > 0))
    shape_conf = max(0.0, min(1.0, edge_density * 4.0))
    if shape_conf < 0.05:
        raise GridDetectionError("shape_mismatch")

    cell_boxes = []
    for r in range(row_count):
        for c in range(col_count):
            cell_boxes.append(
                {
                    "row_1based": r + 1,
                    "col_1based": c + 1,
                    "x0": col_lines[c],
                    "y0": row_lines[r],
                    "x1": col_lines[c + 1],
                    "y1": row_lines[r + 1],
                }
            )

    return GridDetection(
        board_image=warped,
        board_bbox=bbox,
        row_count=row_count,
        col_count=col_count,
        row_lines=row_lines,
        col_lines=col_lines,
        board_confidence=board_conf,
        warp_confidence=warp_conf,
        shape_confidence=shape_conf,
        cell_boxes=cell_boxes,
    )
