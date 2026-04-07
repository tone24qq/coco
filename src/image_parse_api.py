from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from scripts.parse_board_image import parse_image_hybrid
from src.number_position_predictor import predict_number_positions

app = FastAPI(title="Board Parse API")


def _to_tmp(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    out = Path("/tmp") / f"board_api_{file.filename or 'upload'}{suffix}"
    out.write_bytes(file.file.read())
    return str(out)


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
):
    image_path = _to_tmp(image)
    args = Namespace(
        image=image_path,
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
        no_overlay=True,
    )
    payload = parse_image_hybrid(args)
    if not payload.get("contract_passed"):
        raise HTTPException(status_code=422, detail=payload)
    return payload


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
):
    image_path = _to_tmp(image)
    args = Namespace(
        image=image_path,
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
        no_overlay=True,
    )
    payload = parse_image_hybrid(args)
    if not payload.get("contract_passed"):
        raise HTTPException(status_code=422, detail=payload)

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
