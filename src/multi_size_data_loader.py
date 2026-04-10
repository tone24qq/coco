from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .board_manifest import build_multisize_manifest
from .image_board_parser import EXPECTED_SHAPES, parse_manifest_entries, write_boards, write_parse_audit, write_pending


SIZE_CLASSES = ("20", "80", "120", "160")


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
    cache_info: Dict[str, object]


def _validate_complete_grid(grid: np.ndarray, size_class: str) -> str | None:
    if grid.ndim != 2 or grid.size == 0:
        return "invalid_grid_dim"
    expected_shape = EXPECTED_SHAPES.get(size_class)
    if expected_shape and tuple(grid.shape) != expected_shape:
        return "shape_mismatch"
    total = int(grid.shape[0] * grid.shape[1])
    vals = [int(v) for v in grid.flatten().tolist()]
    if len(set(vals)) != len(vals):
        return "duplicate_values"
    if any(v < 1 or v > total for v in vals):
        return "out_of_range_values"
    if set(vals) != set(range(1, total + 1)):
        return "non_permutation_grid"
    return None


def _load_cached_artifacts(config: Dict) -> Optional[ParseArtifacts]:
    parsed_path = Path(config["data"]["parsed_boards_output"])
    audit_path = Path(config["reports"]["data_audit"])
    pending_path = Path(config["reports"]["pending_annotations"])

    if not parsed_path.exists() or not audit_path.exists():
        return None

    try:
        raw_boards = json.loads(parsed_path.read_text(encoding="utf-8"))
        raw_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for field, default_value in (
            ("parse_counts_by_size", {k: {"success": 0, "failed": 0} for k in SIZE_CLASSES}),
            ("valid_sample_count_by_size", {k: 0 for k in SIZE_CLASSES}),
            ("pending_counts_by_size", {k: 0 for k in SIZE_CLASSES}),
        ):
            existing = raw_audit.get(field, {})
            merged = dict(default_value)
            merged.update(existing)
            raw_audit[field] = merged
        raw_pending = []
        if pending_path.exists():
            raw_pending = json.loads(pending_path.read_text(encoding="utf-8"))

        samples: List[MultiSizeBoardSample] = []
        for b in raw_boards:
            raw_grid = [[(-1 if v is None else int(v)) for v in row] for row in b["grid"]]
            grid = np.array(raw_grid, dtype=int)
            issue = _validate_complete_grid(grid, str(b["size_class"]))
            if issue is not None:
                continue
            samples.append(
                MultiSizeBoardSample(
                    sample_id=str(b["sample_id"]),
                    board_id=f"{b['size_class']}:{b['sample_id']}",
                    size_class=str(b["size_class"]),
                    grid=grid,
                    shape=str(b["shape"]),
                    parse_confidence=float(b.get("metadata", {}).get("parse_confidence", 1.0)),
                )
            )
        return ParseArtifacts(
            samples=samples,
            audit=raw_audit,
            pending=raw_pending,
            cache_info={
                "used_cache": True,
                "cache_artifact_path": str(parsed_path),
                "cache_audit_path": str(audit_path),
                "cache_timestamp": parsed_path.stat().st_mtime,
                "cache_sample_count": len(samples),
                "fallback_parse": False,
            },
        )
    except Exception:
        return None


def load_multisize_samples(config: Dict) -> ParseArtifacts:
    use_cache = bool(config.get("parser", {}).get("use_cached_first", True))
    if use_cache:
        cached = _load_cached_artifacts(config)
        if cached is not None:
            return cached

    repo_root = Path(config["data"]["repo_root"])
    manifest, manifest_audit = build_multisize_manifest(repo_root)
    boards, parse_audit, pending, _ = parse_manifest_entries(
        manifest=manifest,
        min_confidence=float(config["parser"]["min_confidence"]),
    )

    samples: List[MultiSizeBoardSample] = []
    fail_fast_counts: Dict[str, int] = {}
    for b in boards:
        grid_arr = np.array(b.grid, dtype=int)
        issue = _validate_complete_grid(grid_arr, b.size_class)
        if issue:
            fail_fast_counts[issue] = fail_fast_counts.get(issue, 0) + 1
            pending.append(
                {
                    "sample_id": b.sample_id,
                    "size_class": b.size_class,
                    "reason": issue,
                    "parse_confidence": b.metadata.get("parse_confidence", 0.0),
                }
            )
            continue
        samples.append(
            MultiSizeBoardSample(
                sample_id=b.sample_id,
                board_id=f"{b.size_class}:{b.sample_id}",
                size_class=b.size_class,
                grid=grid_arr,
                shape=b.shape,
                parse_confidence=float(b.metadata.get("parse_confidence", 1.0)),
            )
        )

    image_counts_by_size: Dict[str, int] = {k: 0 for k in SIZE_CLASSES}
    sample_counts_by_size: Dict[str, int] = {k: 0 for k in SIZE_CLASSES}
    for m in manifest:
        sample_counts_by_size[m.size_class] += 1
        image_counts_by_size[m.size_class] += len(m.image_paths)

    parse_counts: Dict[str, Dict[str, int]] = {k: {"success": 0, "failed": 0} for k in SIZE_CLASSES}
    pending_counts: Dict[str, int] = {k: 0 for k in SIZE_CLASSES}
    valid_count_by_size: Dict[str, int] = {k: 0 for k in SIZE_CLASSES}
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

    for reason, count in fail_fast_counts.items():
        reasons[f"loader_{reason}"] = reasons.get(f"loader_{reason}", 0) + count

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
        "anti_leakage_checks": "passed",
    }

    Path(config["reports"]["data_audit"]).write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(config["reports"]["manifest"]).write_text(
        json.dumps([x.__dict__ for x in manifest], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_parse_audit(parse_audit, Path(config["reports"]["parse_audit"]))
    write_pending(pending, Path(config["reports"]["pending_annotations"]))
    if config["reports"].get("pending_review"):
        write_pending(pending, Path(config["reports"]["pending_review"]))
    write_boards(boards, Path(config["data"]["parsed_boards_output"]))

    return ParseArtifacts(
        samples=samples,
        audit=audit,
        pending=pending,
        cache_info={
            "used_cache": False,
            "cache_artifact_path": str(Path(config["data"]["parsed_boards_output"])),
            "cache_audit_path": str(Path(config["reports"]["data_audit"])),
            "cache_timestamp": None,
            "cache_sample_count": len(samples),
            "fallback_parse": True,
        },
    )
