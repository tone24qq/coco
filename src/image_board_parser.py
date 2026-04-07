from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .board_manifest import ManifestEntry
from .board_structurer import BoardParseResult, structure_board
from .grid_detector import GridDetectionError, detect_grid
from .ticket_specs import SizeClass, get_ticket_spec, validate_page_contract


@dataclass
class ParsedBoard:
    sample_id: str
    size_class: SizeClass
    grid: List[List[Optional[int]]]
    shape: str
    numbers_all: List[int]
    value_to_position: Dict[str, List[Dict[str, int]]]
    metadata: Dict[str, str | int | float | bool | list | dict]


@dataclass
class ParseAuditRow:
    sample_id: str
    size_class: SizeClass
    parse_success: bool
    parse_confidence: float
    shape: str | None
    expected_shape: str
    detected_shape: str | None
    shape_match: bool
    complete_grid: bool
    usable_for_backtest: bool
    pending_review: bool
    hard_fail: bool
    unique_values_ok: bool
    missing_values_ok: bool
    maskable_ok: bool
    ocr_backend: str | None
    hard_fail_reason: str | None
    invalid_reason: str | None


def _merge_pages(image_paths: List[str], size_class: str) -> np.ndarray:
    validate_page_contract(size_class, image_paths)
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    if any(im is None for im in imgs):
        raise ValueError("image_read_failed")
    if len(imgs) == 1:
        return imgs[0]  # type: ignore[return-value]

    widths = [im.shape[1] for im in imgs if im is not None]
    target_w = int(np.median(widths))
    aligned = []
    for im in imgs:
        assert im is not None
        resized = cv2.resize(im, (target_w, int(im.shape[0] * target_w / im.shape[1])))
        aligned.append(resized)
    if aligned[0].shape[1] != aligned[1].shape[1]:
        raise ValueError("page_merge_invalid")
    merged = np.vstack(aligned)
    return merged


def _validate_board(
    grid: List[List[Optional[int]]], expected_shape: tuple[int, int], strict: bool
) -> tuple[bool, bool, bool, bool, str | None]:
    arr = np.array(
        [[x if x is not None else -1 for x in row] for row in grid], dtype=int
    )
    if arr.ndim != 2 or arr.size == 0:
        return False, False, False, False, "invalid_grid_dim"
    shape_match = tuple(arr.shape) == expected_shape
    vals = [int(v) for v in arr.flatten().tolist() if v != -1]
    uniq_ok = len(set(vals)) == len(vals)
    legal = set(range(1, expected_shape[0] * expected_shape[1] + 1))
    miss_ok = set(vals).issubset(legal)
    complete = all(v != -1 for v in arr.flatten().tolist())
    reason = None
    if not shape_match:
        reason = "shape_mismatch"
    elif not (uniq_ok and miss_ok):
        reason = "grid_value_mismatch"
    elif strict and not complete:
        reason = "incomplete_grid"
    return uniq_ok, miss_ok, complete, shape_match, reason


def parse_manifest_entries(
    manifest: List[ManifestEntry],
    min_confidence: float,
    pending_path: Path | None = None,
    strict: bool = False,
) -> tuple[
    List[ParsedBoard],
    List[ParseAuditRow],
    List[Dict[str, object]],
    Dict[str, BoardParseResult],
]:
    boards: List[ParsedBoard] = []
    audits: List[ParseAuditRow] = []
    pending: List[Dict[str, object]] = []
    detailed: Dict[str, BoardParseResult] = {}

    for entry in manifest:
        spec = get_ticket_spec(entry.size_class)
        expected_shape_txt = f"{spec.expected_rows}x{spec.expected_cols}"
        if not entry.valid:
            audits.append(
                ParseAuditRow(
                    entry.sample_id,
                    entry.size_class,
                    False,
                    0.0,
                    None,
                    expected_shape_txt,
                    None,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    None,
                    entry.invalid_reason,
                    entry.invalid_reason,
                )
            )
            pending.append(
                {
                    "sample_id": entry.sample_id,
                    "size_class": entry.size_class,
                    "reason": entry.invalid_reason,
                    "hard_fail_reason": entry.invalid_reason,
                }
            )
            continue

        try:
            gray = _merge_pages(entry.image_paths, entry.size_class)
            det = detect_grid(gray, spec)
            result = structure_board(
                sample_id=entry.sample_id,
                image_path=entry.image_paths[0],
                detection=det,
                spec=spec,
                ticket_type=entry.size_class,
            )
            detailed[entry.sample_id] = result
            uniq_ok, miss_ok, complete, shape_match, reason = _validate_board(
                result.grid, spec.expected_shape, strict
            )
            parse_conf = result.final_parse_confidence
            if parse_conf < min_confidence and reason is None:
                reason = "parse_confidence_low"
            parse_success = reason is None
            pending_review = len(result.pending_cells) > 0
            usable_for_backtest = parse_success and complete and not pending_review
            audits.append(
                ParseAuditRow(
                    sample_id=entry.sample_id,
                    size_class=entry.size_class,
                    parse_success=parse_success,
                    parse_confidence=parse_conf,
                    shape=result.shape,
                    expected_shape=expected_shape_txt,
                    detected_shape=result.shape,
                    shape_match=shape_match,
                    complete_grid=complete,
                    usable_for_backtest=usable_for_backtest,
                    pending_review=pending_review,
                    hard_fail=not parse_success,
                    unique_values_ok=uniq_ok,
                    missing_values_ok=miss_ok,
                    maskable_ok=True,
                    ocr_backend=str(result.parse_diagnostics.get("ocr_backend")),
                    hard_fail_reason=reason,
                    invalid_reason=reason,
                )
            )
            if parse_success:
                boards.append(
                    ParsedBoard(
                        sample_id=entry.sample_id,
                        size_class=entry.size_class,
                        grid=result.grid,
                        shape=result.shape,
                        numbers_all=result.numbers_all,
                        value_to_position=result.value_to_position,
                        metadata={
                            "parse_confidence": parse_conf,
                            "page_count": entry.page_count,
                            "source_folder": entry.source_folder,
                            "low_confidence_cells": result.low_confidence_cells,
                            "expected_shape": expected_shape_txt,
                            "detected_shape": result.shape,
                            "shape_match": shape_match,
                            "complete_grid": complete,
                            "usable_for_backtest": usable_for_backtest,
                            "ocr_backend": result.parse_diagnostics.get("ocr_backend"),
                            "hard_fail_reason": None,
                        },
                    )
                )
            else:
                pending.append(
                    {
                        "sample_id": entry.sample_id,
                        "size_class": entry.size_class,
                        "reason": reason,
                        "hard_fail_reason": reason,
                        "parse_confidence": parse_conf,
                        "low_confidence_cells": result.low_confidence_cells,
                    }
                )
        except (GridDetectionError, ValueError) as exc:
            audits.append(
                ParseAuditRow(
                    entry.sample_id,
                    entry.size_class,
                    False,
                    0.0,
                    None,
                    expected_shape_txt,
                    None,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    None,
                    str(exc),
                    str(exc),
                )
            )
            pending.append(
                {
                    "sample_id": entry.sample_id,
                    "size_class": entry.size_class,
                    "reason": str(exc),
                    "hard_fail_reason": str(exc),
                    "parse_confidence": 0.0,
                }
            )

    if pending_path is not None:
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(
            json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return boards, audits, pending, detailed


def write_boards(boards: List[ParsedBoard], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(x) for x in boards], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_parse_audit(rows: List[ParseAuditRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_pending(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
