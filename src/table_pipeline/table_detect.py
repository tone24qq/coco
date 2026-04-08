from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .grid_refine import repair_line_sequence


class TableDetectError(ValueError):
    pass


@dataclass
class DetectedTable:
    board_bbox: tuple[int, int, int, int]
    row_count: int
    col_count: int
    row_lines: list[int]
    col_lines: list[int]
    cell_boxes: list[tuple[int, int, int, int]]
    diagnostics: dict[str, object]


def _find_peaks(signal: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(signal) - 1):
        if signal[i] < threshold:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif signal[i] > signal[peaks[-1]]:
                peaks[-1] = i
    return peaks


def _line_based_grid(table_bin: np.ndarray) -> tuple[list[int], list[int], dict[str, object]]:
    h, w = table_bin.shape
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 35)))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 35), 1))
    v = cv2.morphologyEx(table_bin, cv2.MORPH_OPEN, vert_kernel)
    hline = cv2.morphologyEx(table_bin, cv2.MORPH_OPEN, hori_kernel)
    v = cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    hline = cv2.dilate(hline, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)

    v_signal = np.sum(v > 0, axis=0).astype(np.float32)
    h_signal = np.sum(hline > 0, axis=1).astype(np.float32)
    v_peaks = _find_peaks(v_signal, float(max(6.0, v_signal.max() * 0.16)), max(4, w // 48))
    h_peaks = _find_peaks(h_signal, float(max(6.0, h_signal.max() * 0.16)), max(4, h // 48))
    if len(v_peaks) < 2 or len(h_peaks) < 2:
        raise TableDetectError("grid_lines_not_found")

    observed_rows = max(1, len(h_peaks) - 1)
    observed_cols = max(1, len(v_peaks) - 1)
    supported = [(5, 4), (10, 8), (12, 10)]
    rows, cols = min(supported, key=lambda rc: abs(rc[0] - observed_rows) + abs(rc[1] - observed_cols))

    row_lines, row_source = repair_line_sequence(h_peaks, h - 1, rows + 1)
    col_lines, col_source = repair_line_sequence(v_peaks, w - 1, cols + 1)
    return row_lines, col_lines, {
        "observed_rows": observed_rows,
        "observed_cols": observed_cols,
        "grid_source": "detected_lines",
        "row_line_source": row_source,
        "col_line_source": col_source,
        "row_lines": [int(x) for x in row_lines],
        "col_lines": [int(x) for x in col_lines],
    }


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
            row_lines, col_lines, diag = _line_based_grid(crop)
        except TableDetectError:
            continue
        rows, cols = len(row_lines) - 1, len(col_lines) - 1
        cell_boxes: list[tuple[int, int, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                x0 = int(x + col_lines[c])
                x1 = int(x + col_lines[c + 1])
                y0 = int(y + row_lines[r])
                y1 = int(y + row_lines[r + 1])
                if x1 - x0 <= 3 or y1 - y0 <= 3:
                    continue
                cell_boxes.append((x0, y0, x1, y1))
        if len(cell_boxes) != rows * cols:
            continue
        tables.append(
            DetectedTable(
                board_bbox=(int(x), int(y), int(w), int(h)),
                row_count=rows,
                col_count=cols,
                row_lines=[int(y + p) for p in row_lines],
                col_lines=[int(x + p) for p in col_lines],
                cell_boxes=cell_boxes,
                diagnostics=diag,
            )
        )
    if not tables:
        raise TableDetectError("table_not_found")
    return tables
