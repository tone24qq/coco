from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class TableDetectError(ValueError):
    pass


@dataclass
class DetectedTable:
    board_bbox: tuple[int, int, int, int]
    row_count: int
    col_count: int
    cell_boxes: list[tuple[int, int, int, int]]


def _find_peaks(signal: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(signal) - 1):
        if signal[i] < threshold:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
    return peaks


def _line_based_grid(table_bin: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = table_bin.shape
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 45)))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 45), 1))
    v = cv2.morphologyEx(table_bin, cv2.MORPH_OPEN, vert_kernel)
    hline = cv2.morphologyEx(table_bin, cv2.MORPH_OPEN, hori_kernel)

    v_signal = np.sum(v > 0, axis=0).astype(np.float32)
    h_signal = np.sum(hline > 0, axis=1).astype(np.float32)
    v_peaks = _find_peaks(v_signal, float(v_signal.max() * 0.18), max(6, w // 40))
    h_peaks = _find_peaks(h_signal, float(h_signal.max() * 0.18), max(6, h // 40))
    if len(v_peaks) < 2 or len(h_peaks) < 2:
        raise TableDetectError("grid_lines_not_found")
    return h_peaks, v_peaks


def detect_tables(binary: np.ndarray) -> list[DetectedTable]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise TableDetectError("table_not_found")
    area_img = float(binary.shape[0] * binary.shape[1])
    tables: list[DetectedTable] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        if (w * h) / area_img < 0.06:
            continue
        crop = binary[y : y + h, x : x + w]
        try:
            row_lines, col_lines = _line_based_grid(crop)
        except TableDetectError:
            continue
        observed_rows = len(row_lines) - 1
        observed_cols = len(col_lines) - 1
        supported = [(5, 4), (10, 8), (12, 10)]
        rows, cols = min(
            supported,
            key=lambda rc: abs(rc[0] - observed_rows) + abs(rc[1] - observed_cols),
        )
        if rows < 2 or cols < 2:
            continue
        row_lines = list(np.linspace(0, h, rows + 1, dtype=int))
        col_lines = list(np.linspace(0, w, cols + 1, dtype=int))
        cell_boxes: list[tuple[int, int, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                x0 = int(x + col_lines[c])
                x1 = int(x + col_lines[c + 1])
                y0 = int(y + row_lines[r])
                y1 = int(y + row_lines[r + 1])
                if x1 - x0 <= 2 or y1 - y0 <= 2:
                    continue
                cell_boxes.append((x0, y0, x1, y1))
        if not cell_boxes:
            continue
        tables.append(
            DetectedTable(
                board_bbox=(int(x), int(y), int(w), int(h)),
                row_count=rows,
                col_count=cols,
                cell_boxes=cell_boxes,
            )
        )
    if not tables:
        raise TableDetectError("table_not_found")
    return tables
