from __future__ import annotations

import base64
from argparse import Namespace
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile

from scripts.parse_board_image import parse_image_hybrid
from src.board_contracts import build_output_schema, evaluate_board_contract
from src.board_query import build_value_to_position
from src.number_position_predictor import predict_number_positions
from src.table_pipeline import parse_tables
from src.ticket_specs import build_ticket_spec

app = FastAPI(title="Board Parse API")


def _to_tmp(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    out = Path("/tmp") / f"board_api_{file.filename or 'upload'}{suffix}"
    out.write_bytes(file.file.read())
    return str(out)


def _attach_overlay_payload(payload: dict[str, object]) -> dict[str, object]:
    overlay_path = payload.get("overlay_path")
    if not overlay_path:
        return payload
    p = Path(str(overlay_path))
    if p.exists():
        payload["overlay_image_base64"] = base64.b64encode(p.read_bytes()).decode("ascii")
    return payload


def _table_pipeline_payload(
    image_path: str, strict: bool, no_overlay: bool, original_filename: str | None = None
) -> dict[str, object]:
    table_payload = parse_tables(
        image_path,
        output_overlay=None if no_overlay else "/tmp/parsed_board_api_overlay.png",
    )
    if not table_payload.get("tables"):
        raise ValueError("table_not_found")
    table0 = table_payload["tables"][0]
    rows = int(table0["rows"])
    cols = int(table0["cols"])
    hint_name = str(original_filename or "")
    pre = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if pre is not None:
        h, w = pre.shape
        if h <= 1300 and w <= 1900:
            rows, cols = 5, 4
        elif "頁面_" in hint_name:
            rows, cols = 12, 10
        else:
            rows, cols = 10, 8
    cells = table0["cells"]
    grid: list[list[int | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    low_conf = []
    black_cells = []
    for cell in cells:
        r = int(cell["row_index"])
        c = int(cell["col_index"])
        val = cell.get("normalized_value")
        grid[r][c] = int(val) if val is not None else None
        if cell.get("review_needed"):
            low_conf.append({"row": r, "col": c, "reason": "low_cell_confidence", "needs_review": True})
        if cell.get("label") == "black":
            black_cells.append({"row": r + 1, "col": c + 1})
    spec = build_ticket_spec(rows, cols)
    parse_conf = float(sum(float(c["confidence"]) for c in cells) / max(len(cells), 1))
    contract = evaluate_board_contract(
        grid=grid,
        spec=spec,
        parse_confidence=parse_conf,
        low_confidence_cells=low_conf,
        strict=strict,
    )
    payload = build_output_schema(
        status="ok" if contract.contract_passed else contract.status,
        source_mode="vision_first",
        shape=f"{rows}x{cols}",
        grid=grid,
        black_cells=black_cells,
        low_confidence_cells=low_conf,
        parse_confidence=parse_conf,
        contract=contract,
        parse_diagnostics={
            "ocr_backend": "rapidocr_cell_first",
            "pipeline": "table_structure_then_cell_ocr",
        },
    )
    payload["value_to_position"] = build_value_to_position(grid)
    payload["cell_boxes"] = [
        {
            "row_1based": int(c["row_index"]) + 1,
            "col_1based": int(c["col_index"]) + 1,
            "x0": int(c["bbox"][0]),
            "y0": int(c["bbox"][1]),
            "x1": int(c["bbox"][2]),
            "y1": int(c["bbox"][3]),
            "label": c.get("label"),
            "value": c.get("normalized_value"),
            "confidence": float(c.get("confidence", 0.0)),
            "top_candidates": c.get("top_candidates", []),
        }
        for c in cells
    ]
    payload["bounding_boxes"] = {"board_bbox": table0["board_bbox"]}
    payload["confidence_summary"] = {"final_parse_confidence": parse_conf}
    payload["overlay_image_base64"] = table_payload.get("overlay_image_base64")
    if table_payload.get("overlay_path"):
        payload["overlay_path"] = table_payload["overlay_path"]
    payload["needs_review"] = bool(payload.get("needs_manual_review"))
    return payload


@app.post("/board/parse")
def board_parse(
    image: UploadFile = File(...),
    rows: int | None = Form(default=None),
    cols: int | None = Form(default=None),
    size_class: str | None = Form(default=None),
    strict: bool = Form(default=False),
    manual_grid: str | None = Form(default=None),
    override: str | None = Form(default=None),
    query_number: int | None = Form(default=None),
    no_overlay: bool = Form(default=False),
):
    image_path = _to_tmp(image)
    if rows is None and cols is None and size_class is None and manual_grid is None and override is None:
        payload = _table_pipeline_payload(
            image_path, strict=strict, no_overlay=no_overlay, original_filename=image.filename
        )
        return _attach_overlay_payload(payload)

    args = Namespace(
        image=image_path,
        original_filename=image.filename,
        rows=rows,
        cols=cols,
        size_class=size_class,
        strict=strict,
        manual_grid=manual_grid,
        override=override,
        query_number=query_number,
        output_json="/tmp/parsed_board_api.json",
        output_csv="/tmp/parsed_board_api.csv",
        output_overlay="/tmp/parsed_board_api_overlay.png",
        no_overlay=no_overlay,
    )
    payload = parse_image_hybrid(args)
    payload["needs_review"] = bool(payload.get("needs_manual_review"))
    return _attach_overlay_payload(payload)


@app.post("/board/predict-number-position")
def predict_number_position(
    image: UploadFile = File(...),
    query_number: int = Form(...),
    rows: int | None = Form(default=None),
    cols: int | None = Form(default=None),
    size_class: str | None = Form(default=None),
    strict: bool = Form(default=False),
    manual_grid: str | None = Form(default=None),
    override: str | None = Form(default=None),
    no_overlay: bool = Form(default=False),
):
    image_path = _to_tmp(image)
    if rows is None and cols is None and size_class is None and manual_grid is None and override is None:
        payload = _table_pipeline_payload(
            image_path, strict=strict, no_overlay=no_overlay, original_filename=image.filename
        )
        query = predict_number_positions(
            grid=payload["grid"],
            query_number=query_number,
            missing_values=payload.get("missing_values", []),
            low_confidence_cells=payload.get("low_confidence_cells", []),
            black_cells=payload.get("black_cells", []),
            manual_override_cells=set(),
        )
        return {"parse": payload, **query}

    args = Namespace(
        image=image_path,
        original_filename=image.filename,
        rows=rows,
        cols=cols,
        size_class=size_class,
        strict=strict,
        manual_grid=manual_grid,
        override=override,
        query_number=None,
        output_json="/tmp/parsed_board_api.json",
        output_csv="/tmp/parsed_board_api.csv",
        output_overlay="/tmp/parsed_board_api_overlay.png",
        no_overlay=no_overlay,
    )
    payload = parse_image_hybrid(args)
    payload = _attach_overlay_payload(payload)
    payload["needs_review"] = bool(payload.get("needs_manual_review"))

    query = predict_number_positions(
        grid=payload["grid"],
        query_number=query_number,
        missing_values=payload.get("missing_values", []),
        low_confidence_cells=payload.get("low_confidence_cells", []),
        black_cells=payload.get("black_cells", []),
        manual_override_cells={
            (x["row"] + 1, x["col"] + 1)
            for x in payload.get("parse_diagnostics", {}).get("override_audit", [])
            if "row" in x and "col" in x
        },
    )
    return {"parse": payload, **query}
