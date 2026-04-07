#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.board_contracts import (
    build_output_schema,
    evaluate_board_contract,
)  # noqa: E402
from src.board_export import (
    grid_to_text,
    write_board_csv,
    write_board_json,
)  # noqa: E402
from src.board_query import find_number_positions  # noqa: E402
from src.board_structurer import BoardParseResult, structure_board  # noqa: E402
from src.grid_detector import GridDetectionError, detect_grid  # noqa: E402
from src.manual_board_input import (
    ManualInputError,
    apply_overrides,
    load_manual_grid,
)  # noqa: E402
from src.ticket_specs import detect_size_class_from_path, get_ticket_spec  # noqa: E402


def _manual_result_from_grid(
    image_path: str, size_class: str, grid: list[list[int | None]]
) -> BoardParseResult:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    black_cells = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v is None:
                black_cells.append({"row": r + 1, "col": c + 1})
    return BoardParseResult(
        sample_id=Path(image_path).stem,
        shape=f"{rows}x{cols}",
        row_count=rows,
        col_count=cols,
        grid=grid,
        cell_confidence=[[1.0 for _ in range(cols)] for _ in range(rows)],
        low_confidence_cells=[],
        parse_confidence=1.0,
        image_path=image_path,
        ticket_type=size_class,
        board_confidence=1.0,
        warp_confidence=1.0,
        shape_confidence=1.0,
        cell_class_confidence_mean=1.0,
        digit_confidence_mean=1.0,
        global_consistency_confidence=1.0,
        final_parse_confidence=1.0,
        numbers_all=sorted([int(v) for row in grid for v in row if v is not None]),
        value_to_position={},
        missing_values=[],
        black_cells=black_cells,
        pending_cells=[],
        parse_diagnostics={"manual_override": True, "ocr_backend": "manual"},
    )


def parse_image_hybrid(args: argparse.Namespace) -> dict[str, object]:
    size_class = args.size_class or detect_size_class_from_path(args.image)
    spec = get_ticket_spec(size_class)

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot_read_image: {args.image}")

    source_mode = "auto"
    base_result: BoardParseResult | None = None
    auto_error: str | None = None
    try:
        det = detect_grid(gray, spec)
        base_result = structure_board(
            sample_id=Path(args.image).stem,
            image_path=args.image,
            detection=det,
            spec=spec,
            ticket_type=size_class,
        )
    except GridDetectionError as exc:
        auto_error = str(exc)

    manual_grid = load_manual_grid(args.manual_grid)
    if base_result is None:
        if manual_grid is None:
            status = auto_error or "needs_manual_review"
            return {
                "status": status,
                "source_mode": "auto",
                "contract_passed": False,
                "needs_manual_review": True,
            }
        base_grid = manual_grid
        source_mode = "manual"
    else:
        base_grid = [row[:] for row in base_result.grid]

    if manual_grid is not None and source_mode == "auto":
        base_grid = manual_grid
        source_mode = "manual"

    if args.override is not None:
        base_grid, override_audit = apply_overrides(base_grid, args.override)
        source_mode = "hybrid" if base_result is not None else "manual"
    else:
        override_audit = []

    final_result = (
        _manual_result_from_grid(args.image, size_class, base_grid)
        if source_mode != "auto"
        else base_result
    )
    assert final_result is not None

    contract = evaluate_board_contract(
        grid=final_result.grid,
        spec=spec,
        parse_confidence=float(final_result.final_parse_confidence),
        low_confidence_cells=final_result.low_confidence_cells,
        min_confidence=0.55,
        max_low_conf_cells=8,
        strict=args.strict,
    )

    status = "ok" if contract.contract_passed else contract.status
    payload = build_output_schema(
        status=status,
        source_mode=source_mode,
        shape=final_result.shape,
        grid=final_result.grid,
        black_cells=final_result.black_cells,
        low_confidence_cells=final_result.low_confidence_cells,
        parse_confidence=float(final_result.final_parse_confidence),
        contract=contract,
        parse_diagnostics={
            **final_result.parse_diagnostics,
            "override_audit": override_audit,
        },
    )

    query_number = getattr(args, "query_number", None)
    if query_number is not None:
        payload["query_result"] = find_number_positions(
            final_result.grid, int(query_number)
        )

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--size-class", choices=["20", "80", "120"], default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--manual-grid", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--query-number", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--output-json", default="reports/parsed_board.json")
    parser.add_argument("--output-csv", default="reports/parsed_board.csv")
    args = parser.parse_args()

    try:
        payload = parse_image_hybrid(args)
    except ManualInputError as exc:
        print(
            json.dumps(
                {"status": str(exc), "needs_manual_review": True}, ensure_ascii=False
            )
        )
        raise SystemExit(2)

    if payload.get("contract_passed"):
        write_board_json(
            _manual_result_from_grid(
                args.image,
                args.size_class or detect_size_class_from_path(args.image),
                payload["grid"],
            ),
            Path(args.output_json),
        )
        write_board_csv(
            _manual_result_from_grid(
                args.image,
                args.size_class or detect_size_class_from_path(args.image),
                payload["grid"],
            ),
            Path(args.output_csv),
        )

    print("=== GRID ===")
    print(grid_to_text(payload.get("grid", [])))
    print("=== NUMBERS ===")
    print(payload.get("numbers_all", []))
    if "query_result" in payload:
        print("=== QUERY ===")
        print(json.dumps(payload["query_result"], ensure_ascii=False))

    print(json.dumps(payload, ensure_ascii=False))
    if not payload.get("contract_passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
