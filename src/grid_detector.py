from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class GridDetection:
    board_image: np.ndarray
    board_bbox: Tuple[int, int, int, int]
    row_count: int
    col_count: int
    row_lines: List[int]
    col_lines: List[int]
    confidence: float


def _cluster_positions(values: np.ndarray, min_gap: int = 8) -> List[int]:
    if values.size == 0:
        return []
    values = np.sort(values)
    groups: List[List[int]] = [[int(values[0])]]
    for v in values[1:]:
        if int(v) - groups[-1][-1] <= min_gap:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return [int(np.mean(g)) for g in groups]


def _coarsen_centers(values: np.ndarray, base_gap: int, max_groups: int) -> List[int]:
    gap = max(base_gap, 4)
    centers = _cluster_positions(values.astype(int), min_gap=gap)
    while len(centers) > max_groups and gap < 200:
        gap = int(gap * 1.35) + 1
        centers = _cluster_positions(values.astype(int), min_gap=gap)
    return centers


def _detect_board_region(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    area_img = float(h * w)
    if not contours:
        return gray, (0, 0, w, h), 0.25

    best = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(best)
    area_ratio = (bw * bh) / max(area_img, 1.0)
    # 80/120 images often have weak outer contour; fallback to full image instead of fail-fast.
    if area_ratio < 0.05:
        return gray, (0, 0, w, h), 0.35
    board = gray[y : y + bh, x : x + bw]
    conf = min(1.0, max(0.0, area_ratio))
    return board, (x, y, bw, bh), conf


def _infer_lines_from_components(bin_img: np.ndarray) -> Tuple[List[int], List[int], float]:
    n, _labels, stats, cent = cv2.connectedComponentsWithStats(bin_img, 8)
    if n <= 1:
        return [], [], 0.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = (areas > 20) & (areas < 5000) & (heights > 6) & (widths > 3)
    if np.sum(valid) < 20:
        return [], [], 0.0

    cx = cent[1:, 0][valid]
    cy = cent[1:, 1][valid]
    h_med = float(np.median(heights[valid]))
    w_med = float(np.median(widths[valid]))
    row_centers = _coarsen_centers(cy, base_gap=max(6, int(h_med * 0.9)), max_groups=20)
    col_centers = _coarsen_centers(cx, base_gap=max(8, int(w_med * 1.6)), max_groups=20)
    if len(row_centers) < 2 or len(col_centers) < 2:
        return [], [], 0.0

    def centers_to_lines(cs: List[int], limit: int) -> List[int]:
        cs = sorted(cs)
        mids = [0]
        for i in range(len(cs) - 1):
            mids.append((cs[i] + cs[i + 1]) // 2)
        mids.append(limit - 1)
        return mids

    row_lines = centers_to_lines(row_centers, bin_img.shape[0])
    col_lines = centers_to_lines(col_centers, bin_img.shape[1])
    conf = min(0.75, 0.3 + 0.01 * len(row_centers) + 0.01 * len(col_centers))
    return row_lines, col_lines, conf


def detect_grid(gray: np.ndarray) -> GridDetection:
    board, bbox, board_conf = _detect_board_region(gray)
    bin_img = cv2.adaptiveThreshold(
        board,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    h, w = board.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 20)))
    hlines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, hk)
    vlines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vk)

    row_strength = np.sum(hlines > 0, axis=1)
    col_strength = np.sum(vlines > 0, axis=0)
    row_idx = np.where(row_strength > 0.35 * np.max(row_strength))[0] if np.max(row_strength) > 0 else np.array([])
    col_idx = np.where(col_strength > 0.35 * np.max(col_strength))[0] if np.max(col_strength) > 0 else np.array([])
    row_lines = _cluster_positions(row_idx, min_gap=max(4, h // 120))
    col_lines = _cluster_positions(col_idx, min_gap=max(4, w // 120))
    line_conf = 0.7

    if len(row_lines) < 3 or len(col_lines) < 3:
        row_lines, col_lines, line_conf = _infer_lines_from_components(bin_img)

    if len(row_lines) < 3 or len(col_lines) < 3:
        raise ValueError("grid_lines_unstable")

    row_lines = sorted(set([0] + row_lines + [h - 1]))
    col_lines = sorted(set([0] + col_lines + [w - 1]))
    row_lines = [x for i, x in enumerate(row_lines) if i == 0 or (x - row_lines[i - 1]) > 6]
    col_lines = [x for i, x in enumerate(col_lines) if i == 0 or (x - col_lines[i - 1]) > 6]

    row_count = len(row_lines) - 1
    col_count = len(col_lines) - 1
    if row_count < 2 or col_count < 2:
        raise ValueError("grid_shape_invalid")

    cell_h = np.diff(np.array(row_lines))
    cell_w = np.diff(np.array(col_lines))
    var_penalty = float(np.var(cell_h) / max(np.mean(cell_h), 1.0) + np.var(cell_w) / max(np.mean(cell_w), 1.0))
    conf = max(0.0, min(1.0, (0.5 * board_conf + 0.5 * line_conf) * (1.0 / (1.0 + 0.002 * var_penalty))))

    return GridDetection(
        board_image=board,
        board_bbox=bbox,
        row_count=row_count,
        col_count=col_count,
        row_lines=row_lines,
        col_lines=col_lines,
        confidence=conf,
    )
