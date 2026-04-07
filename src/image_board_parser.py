from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import cv2
import numpy as np

from .board_manifest import ManifestEntry
from .board_structurer import BoardParseResult, structure_board
from .grid_detector import detect_grid

SizeClass = Literal["20", "80", "120"]


@dataclass
class ParsedBoard:
    sample_id: str
    size_class: SizeClass
    grid: List[List[Optional[int]]]
    shape: str
    numbers_all: List[int]
    metadata: Dict[str, str | int | float | bool | list]


@dataclass
class ParseAuditRow:
    sample_id: str
    size_class: SizeClass
    parse_success: bool
    parse_confidence: float
    shape: str | None
    unique_values_ok: bool
    missing_values_ok: bool
    maskable_ok: bool
    invalid_reason: str | None


def _merge_pages(image_paths: List[str]) -> np.ndarray:
    imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    if any(im is None for im in imgs):
        raise ValueError("image_read_failed")
    if len(imgs) == 1:
        return imgs[0]  # type: ignore[return-value]
    widths = [im.shape[1] for im in imgs if im is not None]
    target_w = int(np.median(widths))
    resized = [cv2.resize(im, (target_w, int(im.shape[0] * target_w / im.shape[1]))) for im in imgs if im is not None]
    return np.vstack(resized)


def _validate_board(grid: List[List[Optional[int]]]) -> tuple[bool, bool, bool, str | None]:
    arr = np.array([[x if x is not None else -1 for x in row] for row in grid], dtype=int)
    if arr.ndim != 2 or arr.size == 0:
        return False, False, False, "invalid_grid_dim"
    total = arr.shape[0] * arr.shape[1]
    vals = [int(v) for v in arr.flatten().tolist() if v != -1]
    uniq_ok = len(set(vals)) == len(vals)
    miss_ok = set(vals).issubset(set(range(1, total + 1)))
    maskable_ok = total >= 2
    reason = None if (uniq_ok and miss_ok and maskable_ok) else "grid_value_mismatch"
    return uniq_ok, miss_ok, maskable_ok, reason


def parse_manifest_entries(
    manifest: List[ManifestEntry],
    min_confidence: float,
    pending_path: Path | None = None,
) -> tuple[List[ParsedBoard], List[ParseAuditRow], List[Dict[str, object]], Dict[str, BoardParseResult]]:
    boards: List[ParsedBoard] = []
    audits: List[ParseAuditRow] = []
    pending: List[Dict[str, object]] = []
    detailed: Dict[str, BoardParseResult] = {}

    for entry in manifest:
        if not entry.valid:
            audits.append(
                ParseAuditRow(
                    sample_id=entry.sample_id,
                    size_class=entry.size_class,
                    parse_success=False,
                    parse_confidence=0.0,
                    shape=None,
                    unique_values_ok=False,
                    missing_values_ok=False,
                    maskable_ok=False,
                    invalid_reason=entry.invalid_reason or "invalid_manifest_entry",
                )
            )
            pending.append(
                {
                    "sample_id": entry.sample_id,
                    "size_class": entry.size_class,
                    "reason": entry.invalid_reason,
                }
            )
            continue

        try:
            gray = _merge_pages(entry.image_paths)
            det = detect_grid(gray)
            result = structure_board(
                sample_id=entry.sample_id,
                image_path=entry.image_paths[0],
                detection=det,
                ticket_type=entry.size_class,
            )
            detailed[entry.sample_id] = result
            uniq_ok, miss_ok, maskable_ok, reason = _validate_board(result.grid)
            if result.parse_confidence < min_confidence:
                reason = "parse_confidence_low"
            parse_success = reason is None
            audits.append(
                ParseAuditRow(
                    sample_id=entry.sample_id,
                    size_class=entry.size_class,
                    parse_success=parse_success,
                    parse_confidence=result.parse_confidence,
                    shape=result.shape,
                    unique_values_ok=uniq_ok,
                    missing_values_ok=miss_ok,
                    maskable_ok=maskable_ok,
                    invalid_reason=reason,
                )
            )
            if parse_success:
                nums = sorted([int(v) for row in result.grid for v in row if v is not None])
                boards.append(
                    ParsedBoard(
                        sample_id=entry.sample_id,
                        size_class=entry.size_class,
                        grid=result.grid,
                        shape=result.shape,
                        numbers_all=nums,
                        metadata={
                            "parse_confidence": result.parse_confidence,
                            "page_count": entry.page_count,
                            "source_folder": entry.source_folder,
                            "low_confidence_cells": result.low_confidence_cells,
                        },
                    )
                )
            else:
                pending.append(
                    {
                        "sample_id": entry.sample_id,
                        "size_class": entry.size_class,
                        "reason": reason,
                        "parse_confidence": result.parse_confidence,
                        "low_confidence_cells": result.low_confidence_cells,
                    }
                )
        except Exception as exc:  # fail-fast per sample
            audits.append(
                ParseAuditRow(
                    sample_id=entry.sample_id,
                    size_class=entry.size_class,
                    parse_success=False,
                    parse_confidence=0.0,
                    shape=None,
                    unique_values_ok=False,
                    missing_values_ok=False,
                    maskable_ok=False,
                    invalid_reason=str(exc),
                )
            )
            pending.append(
                {
                    "sample_id": entry.sample_id,
                    "size_class": entry.size_class,
                    "reason": str(exc),
                    "parse_confidence": 0.0,
                }
            )

    if pending_path is not None:
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")

    return boards, audits, pending, detailed


def write_boards(boards: List[ParsedBoard], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(x) for x in boards], indent=2, ensure_ascii=False), encoding="utf-8")


def write_parse_audit(rows: List[ParseAuditRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False), encoding="utf-8")


def write_pending(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
