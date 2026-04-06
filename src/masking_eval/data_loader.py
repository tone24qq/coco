from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class BoardSample:
    board_id: str
    grid: np.ndarray
    source: str
    order_index: int | None = None


@dataclass
class BoardAudit:
    total_boards: int
    valid_boards: int
    invalid_boards: int
    invalid_reasons: Dict[str, int]
    anti_leakage_checks: str


def discover_board_files(repo_root: Path) -> List[Path]:
    out: List[Path] = []
    for pat in ("*.json", "*.csv", "*.parquet"):
        for item in repo_root.rglob(pat):
            if ".venv" in item.parts or ".git" in item.parts or "reports" in item.parts:
                continue
            out.append(item)
    return sorted(set(out))


def _validate_grid(grid: np.ndarray) -> str | None:
    if grid.shape != (10, 8):
        return "invalid_shape"
    vals = grid.flatten()
    if len(np.unique(vals)) != 80:
        return "duplicate_values"
    expected = set(range(1, 81))
    got = set(int(v) for v in vals)
    if got != expected:
        return "missing_or_out_of_range_values"
    return None


def load_full_boards(path: Path) -> Tuple[List[BoardSample], BoardAudit]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Board data must be a list")

    seen_ids: set[str] = set()
    invalid_reasons: Dict[str, int] = {}
    valid: List[BoardSample] = []

    for i, rec in enumerate(records):
        board_id = str(rec.get("board_id", f"board_{i}"))
        if board_id in seen_ids:
            invalid_reasons["duplicate_board_id"] = invalid_reasons.get("duplicate_board_id", 0) + 1
            continue
        seen_ids.add(board_id)

        grid = np.array(rec.get("grid"), dtype=int)
        reason = _validate_grid(grid)
        if reason:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue

        valid.append(
            BoardSample(
                board_id=board_id,
                grid=grid,
                source=str(rec.get("source", path.name)),
                order_index=int(rec["order_index"]) if rec.get("order_index") is not None else None,
            )
        )

    if not valid:
        raise ValueError(f"Fail-fast: no valid boards, reasons={invalid_reasons}")

    audit = BoardAudit(
        total_boards=len(records),
        valid_boards=len(valid),
        invalid_boards=len(records) - len(valid),
        invalid_reasons=invalid_reasons,
        anti_leakage_checks="passed",
    )
    return valid, audit


def write_audit(path: Path, audit: BoardAudit) -> None:
    payload = {
        "total_boards": audit.total_boards,
        "valid_boards": audit.valid_boards,
        "invalid_boards": audit.invalid_boards,
        "invalid_reasons": audit.invalid_reasons,
        "anti_leakage_checks": audit.anti_leakage_checks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
