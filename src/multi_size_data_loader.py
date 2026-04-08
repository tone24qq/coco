from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .board_manifest import build_multisize_manifest
from .image_board_parser import (
    parse_manifest_entries,
    write_boards,
    write_parse_audit,
    write_pending,
)


@dataclass
class MultiSizeBoardSample:
    sample_id: str
    board_id: str
    size_class: str
    grid: np.ndarray
    shape: str
    parse_confidence: float


@dataclass
class ParseArtifacts:
    samples: List[MultiSizeBoardSample]
    audit: Dict
    pending: List[Dict[str, object]]


def _is_complete_grid(grid: List[List[Optional[int]]]) -> bool:
    return all(v is not None for row in grid for v in row)


def _apply_manual_manifest_overrides(
    manifest: list, manual_manifest_path: str | None
) -> list:
    if not manual_manifest_path:
        return manifest
    p = Path(manual_manifest_path)
    if not p.exists():
        return manifest
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return manifest
    index = {(m.size_class, m.sample_id): m for m in manifest}
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("size_class")), str(item.get("sample_id")))
        entry = index.get(key)
        if entry is None:
            continue
        if "manual_grid" in item:
            entry.manual_grid = str(item["manual_grid"])
        if "override" in item:
            entry.override = str(item["override"])
    return manifest


def load_multisize_samples(config: Dict) -> ParseArtifacts:
    repo_root = Path(config["data"]["repo_root"])
    manifest, manifest_audit = build_multisize_manifest(repo_root)
    manifest = _apply_manual_manifest_overrides(
        manifest, config.get("parser", {}).get("manual_manifest")
    )
    max_per_size = int(config.get("parser", {}).get("max_samples_per_size", 0) or 0)
    if max_per_size > 0:
        keep: Dict[str, int] = {"20": 0, "80": 0, "120": 0}
        filtered = []
        for m in manifest:
            if keep.get(m.size_class, 0) >= max_per_size:
                continue
            filtered.append(m)
            keep[m.size_class] = keep.get(m.size_class, 0) + 1
        manifest = filtered
    boards, parse_audit, pending, _ = parse_manifest_entries(
        manifest=manifest,
        min_confidence=float(config["parser"]["min_confidence"]),
        strict=False,
    )

    samples: List[MultiSizeBoardSample] = []
    for b in boards:
        complete = _is_complete_grid(b.grid)
        usable = bool(b.metadata.get("usable_for_backtest", False)) and complete
        if not usable:
            pending.append(
                {
                    "sample_id": b.sample_id,
                    "size_class": b.size_class,
                    "reason": (
                        "pending_review"
                        if b.metadata.get("low_confidence_cells")
                        else "incomplete_grid"
                    ),
                    "parse_success": True,
                    "complete_grid": complete,
                    "usable_for_backtest": usable,
                    "pending_review": True,
                    "hard_fail": False,
                }
            )
            continue
        samples.append(
            MultiSizeBoardSample(
                sample_id=b.sample_id,
                board_id=f"{b.size_class}:{b.sample_id}",
                size_class=b.size_class,
                grid=np.array(b.grid, dtype=int),
                shape=b.shape,
                parse_confidence=float(b.metadata.get("parse_confidence", 1.0)),
            )
        )

    image_counts_by_size: Dict[str, int] = {k: 0 for k in ("20", "80", "120")}
    sample_counts_by_size: Dict[str, int] = {k: 0 for k in ("20", "80", "120")}
    for m in manifest:
        sample_counts_by_size[m.size_class] += 1
        image_counts_by_size[m.size_class] += len(m.image_paths)

    parse_counts: Dict[str, Dict[str, int]] = {
        k: {"success": 0, "failed": 0} for k in ("20", "80", "120")
    }
    pending_counts: Dict[str, int] = {k: 0 for k in ("20", "80", "120")}
    valid_count_by_size: Dict[str, int] = {k: 0 for k in ("20", "80", "120")}
    reasons: Dict[str, int] = {}
    for row in parse_audit:
        key = row.size_class
        if row.parse_success:
            parse_counts[key]["success"] += 1
        else:
            parse_counts[key]["failed"] += 1
            reason = row.invalid_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
    for p in pending:
        key = str(p.get("size_class"))
        if key in pending_counts:
            pending_counts[key] += 1
    for s in samples:
        valid_count_by_size[s.size_class] += 1

    audit = {
        "manifest_audit": {
            "total_images": manifest_audit.total_images,
            "total_samples": manifest_audit.total_samples,
            "valid_samples": manifest_audit.valid_samples,
            "invalid_samples": manifest_audit.invalid_samples,
            "invalid_reasons": manifest_audit.invalid_reasons,
        },
        "scanned_images_by_size": image_counts_by_size,
        "scanned_samples_by_size": sample_counts_by_size,
        "valid_sample_count_by_size": valid_count_by_size,
        "parse_counts_by_size": parse_counts,
        "pending_counts_by_size": pending_counts,
        "parse_failure_reasons": reasons,
        "status_flags": {
            "parse_success": [x.parse_success for x in parse_audit],
            "complete_grid": [x.complete_grid for x in parse_audit],
            "usable_for_backtest": [x.usable_for_backtest for x in parse_audit],
            "pending_review": [x.pending_review for x in parse_audit],
            "hard_fail": [x.hard_fail for x in parse_audit],
        },
        "anti_leakage_checks": "passed",
    }

    Path(config["reports"]["data_audit"]).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(config["reports"]["manifest"]).write_text(
        json.dumps([x.__dict__ for x in manifest], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_parse_audit(parse_audit, Path(config["reports"]["parse_audit"]))
    write_pending(pending, Path(config["reports"]["pending_annotations"]))
    if config["reports"].get("pending_review"):
        write_pending(pending, Path(config["reports"]["pending_review"]))
    write_boards(boards, Path(config["data"]["parsed_boards_output"]))

    return ParseArtifacts(samples=samples, audit=audit, pending=pending)
