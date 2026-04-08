from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cell_ocr import ocr_cell
from .global_decode import decode_with_constraints
from .postprocess import build_pipeline_record, render_overlay
from .preprocess import preprocess_image
from .schemas import CellRecord, TableRecord
from .table_detect import detect_tables


class TablePipelineError(ValueError):
    pass


def parse_tables(image_path: str, output_overlay: str | None = None) -> dict[str, object]:
    pre = preprocess_image(image_path)
    detected = detect_tables(pre.binary)
    tables: list[TableRecord] = []
    for ti, dt in enumerate(detected):
        cells: list[CellRecord] = []
        max_value = dt.row_count * dt.col_count
        candidates: dict[tuple[int, int], list[dict[str, float | int]]] = {}
        low_conf_cells: list[dict[str, object]] = []
        ocr_backends: set[str] = set()
        for idx, box in enumerate(dt.cell_boxes):
            r = idx // dt.col_count
            c = idx % dt.col_count
            rec = ocr_cell(pre.enhanced, box, max_value=max_value)
            candidates[(r, c)] = list(rec.get("top_candidates", []))
            ocr_backends.add(str(rec.get("ocr_backend", "unknown")))
            cells.append(
                CellRecord(
                    row_index=r,
                    col_index=c,
                    bbox=box,
                    text=str(rec["text"]),
                    confidence=float(rec["confidence"]),
                    is_numeric=bool(rec["is_numeric"]),
                    normalized_value=rec["normalized_value"],
                    review_needed=bool(rec["review_needed"]),
                    label=str(rec["label"]),
                    top_candidates=list(rec["top_candidates"]),
                )
            )

        decoded = decode_with_constraints(dt.row_count, dt.col_count, candidates)
        low_conf_cells.extend(decoded.low_confidence_cells)
        for cell in cells:
            val = decoded.grid[cell.row_index][cell.col_index]
            cell.normalized_value = val
            cell.text = str(val) if val is not None else ""
            if val is None:
                cell.review_needed = True

        tables.append(
            TableRecord(
                table_index=ti,
                board_bbox=dt.board_bbox,
                rows=dt.row_count,
                cols=dt.col_count,
                cells=cells,
                diagnostics={
                    **dt.diagnostics,
                    "preprocess": pre.diagnostics,
                    "decode_backend": decoded.decode_backend,
                    "ocr_backends": sorted(ocr_backends),
                    "fallback_used": False,
                    "fallback_reason": None,
                    "low_confidence_cells": low_conf_cells,
                },
            )
        )

    payload = build_pipeline_record(tables).to_dict()
    overlay_path, overlay_b64 = render_overlay(pre.image_bgr, tables, output_overlay)
    payload["overlay_path"] = overlay_path
    payload["overlay_image_base64"] = overlay_b64
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", default="reports/table_pipeline_output.json")
    parser.add_argument("--output-overlay", default="reports/table_pipeline_overlay.png")
    args = parser.parse_args()

    payload = parse_tables(args.image, output_overlay=args.output_overlay)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
