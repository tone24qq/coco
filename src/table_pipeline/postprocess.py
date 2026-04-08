from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .schemas import PipelineRecord, TableRecord


def build_pipeline_record(tables: list[TableRecord]) -> PipelineRecord:
    return PipelineRecord(is_table_document=len(tables) > 0, tables=tables)


def render_overlay(
    image_bgr: np.ndarray,
    table_records: list[TableRecord],
    output_path: str | None = None,
) -> tuple[str | None, str | None]:
    vis = image_bgr.copy()
    for table in table_records:
        x, y, w, h = table.board_bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for cell in table.cells:
            x0, y0, x1, y1 = cell.bbox
            color = (0, 0, 255) if cell.review_needed else (255, 120, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
            txt = f"r{cell.row_index+1}c{cell.col_index+1}:{cell.text or '?'} {cell.confidence:.2f}"
            cv2.putText(
                vis,
                txt,
                (x0 + 2, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 255),
                1,
            )
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), vis)
        return str(p), base64.b64encode(p.read_bytes()).decode("ascii")

    ok, encoded = cv2.imencode(".png", vis)
    if not ok:
        return None, None
    return None, base64.b64encode(encoded.tobytes()).decode("ascii")
