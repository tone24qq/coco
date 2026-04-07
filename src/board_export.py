from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import cv2

from .board_structurer import BoardParseResult
from .grid_detector import GridDetection


def grid_to_text(grid: List[List[Optional[int]]]) -> str:
    rows = []
    for row in grid:
        rows.append(" ".join([f"{v:>2}" if v is not None else "□ " for v in row]))
    return "\n".join(rows)


def write_board_json(result: BoardParseResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def write_board_csv(result: BoardParseResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(result.grid)


def write_overlay(result: BoardParseResult, det: GridDetection, image_path: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("overlay_image_read_failed")
    x, y, bw, bh = det.board_bbox
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    for r in range(result.row_count):
        for c in range(result.col_count):
            yy0, yy1 = det.row_lines[r], det.row_lines[r + 1]
            xx0, xx1 = det.col_lines[c], det.col_lines[c + 1]
            p0 = (x + xx0, y + yy0)
            p1 = (x + xx1, y + yy1)
            cv2.rectangle(vis, p0, p1, (255, 120, 0), 1)
            txt = str(result.grid[r][c]) if result.grid[r][c] is not None else "□"
            cv2.putText(vis, txt, (p0[0] + 3, p0[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(str(output_path), vis)
