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
    write_overlay,
)  # noqa: E402
from src.board_query import find_number_positions  # noqa: E402
from src.board_structurer import BoardParseResult, structure_board  # noqa: E402
from src.grid_detector import (
    GridDetection,
    GridDetectionError,
    detect_grid,
    infer_grid_shape,
)  # noqa: E402
from src.manual_board_input import (
    ManualInputError,
    apply_overrides,
    load_manual_grid_with_states,
)  # noqa: E402
from src.ticket_specs import TicketSpec, resolve_ticket_spec  # noqa: E402
from src.ticket_specs import TICKET_SPECS, get_ticket_spec  # noqa: E402


def _manual_result_from_grid(
    image_path: str,
    spec: TicketSpec,
    grid: list[list[int | None]],
    manual_states: list[list[str]] | None = None,
) -> BoardParseResult:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    black_cells = []
    cell_boxes = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            state = (
                manual_states[r][c]
                if manual_states and r < len(manual_states) and c < len(manual_states[r])
                else ("empty" if v is None else "confirmed_number")
            )
            if v is None and state == "black":
                black_cells.append({"row": r + 1, "col": c + 1})
            cell_boxes.append(
                {
                    "row_1based": r + 1,
                    "col_1based": c + 1,
                    "x0": 0,
                    "y0": 0,
                    "x1": 0,
                    "y1": 0,
                    "label": "manual",
                    "value": v,
                    "confidence": 1.0,
                    "manual_state": state,
                }
            )
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
        ticket_type=spec.size_class,
        board_confidence=1.0,
        warp_confidence=1.0,
        shape_confidence=1.0,
        cell_class_confidence_mean=1.0,
        digit_confidence_mean=1.0,
        global_consistency_confidence=1.0,
        final_parse_confidence=1.0,
        numbers_all=sorted([int(v) for row in grid for v in row if v is not None]),
        value_to_position={},
        missing_values=sorted(
            list(
                spec.legal_values
                - {int(v) for row in grid for v in row if v is not None}
            )
        ),
        black_cells=black_cells,
        pending_cells=[],
        parse_diagnostics={"manual_override": True, "ocr_backend": "manual"},
        cell_boxes=cell_boxes,
    )


def parse_image_hybrid(args: argparse.Namespace) -> dict[str, object]:
    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot_read_image:{args.image}")
    manual_grid, manual_states = load_manual_grid_with_states(
        getattr(args, "manual_grid", None)
    )
    detection_pref: GridDetection | None = None
    base_result_pref: BoardParseResult | None = None
    if any(
        getattr(args, key, None) is not None for key in ("size_class", "rows", "cols")
    ):
        spec = resolve_ticket_spec(
            size_class=getattr(args, "size_class", None),
            rows=getattr(args, "rows", None),
            cols=getattr(args, "cols", None),
            image_path=args.image,
        )
    else:
        img_h, img_w = gray.shape[:2]
        original_filename = str(getattr(args, "original_filename", "") or "")
        inferred_shape: tuple[int, int] | None = None
        infer_conf = 0.0
        try:
            inferred_shape, infer_conf = infer_grid_shape(gray)
        except GridDetectionError:
            inferred_shape = None
            infer_conf = 0.0
        size_class_candidates = list(TICKET_SPECS.keys())
        preferred_size = None
        if img_h <= 1300 and img_w <= 1900:
            spec = get_ticket_spec("20")
            try:
                detection_pref = detect_grid(gray, spec)
            except GridDetectionError:
                detection_pref = None
            candidates: list[
                tuple[float, TicketSpec, GridDetection, BoardParseResult | None]
            ] = []
            if detection_pref is not None and manual_grid is None:
                try:
                    base_result_pref = structure_board(
                        sample_id=Path(args.image).stem,
                        image_path=args.image,
                        detection=detection_pref,
                        spec=spec,
                        ticket_type=spec.size_class,
                    )
                except Exception:
                    base_result_pref = None
            # short-circuit for 20-shape paper format
            if detection_pref is not None:
                candidates = [(1.0, spec, detection_pref, base_result_pref)]
                size_class_candidates = []
        else:
            is_multipage_hint = "頁面_" in original_filename or "頁面_" in str(args.image)
            preferred_size = "120" if is_multipage_hint else "80"
            if is_multipage_hint:
                size_class_candidates = ["120"]
            else:
                size_class_candidates = ["80"]
            candidates = []
        for size_class in size_class_candidates:
            candidate_spec = get_ticket_spec(size_class)
            try:
                candidate_det = detect_grid(gray, candidate_spec)
            except GridDetectionError:
                continue
            candidate_result: BoardParseResult | None = None
            if manual_grid is None:
                try:
                    candidate_result = structure_board(
                        sample_id=Path(args.image).stem,
                        image_path=args.image,
                        detection=candidate_det,
                        spec=candidate_spec,
                        ticket_type=candidate_spec.size_class,
                    )
                except Exception:
                    candidate_result = None
            if candidate_result is not None:
                total = max(
                    1, candidate_spec.expected_rows * candidate_spec.expected_cols
                )
                numbers = [int(v) for row in candidate_result.grid for v in row if v is not None]
                complete_ratio = len(numbers) / float(total)
                dup_count = min(
                    len(candidate_result.parse_diagnostics.get("duplicates", [])), total
                )
                illegal_count = len([v for v in numbers if v < 1 or v > total])
                low_count = min(len(candidate_result.low_confidence_cells), total)
                score = (
                    0.35 * float(candidate_result.final_parse_confidence)
                    + 0.50 * complete_ratio
                    - 0.25 * (dup_count / float(total))
                    - 0.35 * (illegal_count / float(total))
                    - 0.20 * (low_count / float(total))
                )
            else:
                score = (
                    0.4 * float(candidate_det.board_confidence)
                    + 0.3 * float(candidate_det.warp_confidence)
                    + 0.3 * float(candidate_det.shape_confidence)
                )
            if preferred_size is not None and size_class == preferred_size:
                score += 0.03
            candidates.append((score, candidate_spec, candidate_det, candidate_result))
        if not candidates:
            raise GridDetectionError("shape_resolution_failed")
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, spec, detection_pref, base_result_pref = candidates[0]
    override_path = getattr(args, "override", None)
    source_mode = "auto"
    base_result: BoardParseResult | None = None
    detection: GridDetection | None = None
    auto_error: str | None = None

    # manual-first: if manual-grid is provided, do not depend on auto parsing as the main source.
    if manual_grid is not None:
        source_mode = "manual"
        base_grid = manual_grid
        # optional detection for bbox/overlay only; never blocks manual flow
        detection = detection_pref
        if detection is None:
            try:
                detection = detect_grid(gray, spec)
            except GridDetectionError:
                detection = None
    else:
        try:
            detection = detection_pref or detect_grid(gray, spec)
            base_result = base_result_pref
            if base_result is None:
                base_result = structure_board(
                    sample_id=Path(args.image).stem,
                    image_path=args.image,
                    detection=detection,
                    spec=spec,
                    ticket_type=spec.size_class,
                )
            base_grid = [row[:] for row in base_result.grid]
        except GridDetectionError as exc:
            auto_error = str(exc)
            if override_path is None:
                return {
                    "status": auto_error or "needs_manual_review",
                    "source_mode": "auto",
                    "contract_passed": False,
                    "needs_manual_review": True,
                }
            base_grid = [
                [None for _ in range(spec.expected_cols)]
                for _ in range(spec.expected_rows)
            ]
            manual_states = [
                ["unknown" for _ in range(spec.expected_cols)]
                for _ in range(spec.expected_rows)
            ]
            source_mode = "manual"

    override_audit = []
    if override_path:
        base_grid, override_audit = apply_overrides(base_grid, override_path)
        source_mode = "hybrid" if source_mode in ("manual", "auto") else source_mode

    final_result = (
        _manual_result_from_grid(args.image, spec, base_grid, manual_states=manual_states)
        if source_mode in ("manual", "hybrid")
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
        strict=getattr(args, "strict", False),
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
            "board_bbox": detection.board_bbox if detection else None,
            "parse_policy": "manual_first",
            "manual_cell_states": manual_states,
        },
    )
    payload["cell_boxes"] = final_result.cell_boxes
    payload["bounding_boxes"] = {
        "board_bbox": detection.board_bbox if detection else None
    }
    payload["confidence_summary"] = {
        "board_confidence": float(final_result.board_confidence),
        "warp_confidence": float(final_result.warp_confidence),
        "shape_confidence": float(final_result.shape_confidence),
        "cell_class_confidence_mean": float(final_result.cell_class_confidence_mean),
        "digit_confidence_mean": float(final_result.digit_confidence_mean),
        "global_consistency_confidence": float(
            final_result.global_consistency_confidence
        ),
        "final_parse_confidence": float(final_result.final_parse_confidence),
    }

    query_number = getattr(args, "query_number", None)
    if query_number is not None:
        payload["query_result"] = find_number_positions(
            final_result.grid, int(query_number)
        )

    if payload.get("contract_passed"):
        out_json = getattr(args, "output_json", "reports/parsed_board.json")
        out_csv = getattr(args, "output_csv", "reports/parsed_board.csv")
        write_board_json(final_result, Path(out_json))
        write_board_csv(final_result, Path(out_csv))
    output_overlay = getattr(
        args, "output_overlay", "reports/parsed_board_overlay.png"
    )
    if (
        not getattr(args, "no_overlay", False)
        and output_overlay
        and detection is not None
    ):
        write_overlay(final_result, detection, args.image, Path(output_overlay))
        payload["overlay_path"] = output_overlay

    payload = json.loads(json.dumps(payload, default=lambda o: int(o)))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--size-class", choices=["20", "80", "120"], default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--manual-grid", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--query-number", type=int, default=None)
    parser.add_argument("--output-json", default="reports/parsed_board.json")
    parser.add_argument("--output-csv", default="reports/parsed_board.csv")
    parser.add_argument("--output-overlay", default="reports/parsed_board_overlay.png")
    parser.add_argument("--no-overlay", action="store_true")
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

    print("=== GRID ===")
    print(grid_to_text(payload.get("grid", [])))
    print("=== NUMBERS ===")
    print(payload.get("numbers_all", []))
    if "query_result" in payload:
        print("=== QUERY ===")
        print(json.dumps(payload["query_result"], ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False, default=lambda o: int(o)))
    if not payload.get("contract_passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
