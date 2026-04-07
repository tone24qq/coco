from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ManualInputError(ValueError):
    pass


def _read_json(path: str | None) -> object | None:
    if path is None:
        return None
    raw = str(path).strip()
    if raw.startswith("{") or raw.startswith("["):
        return json.loads(raw)
    return json.loads(Path(raw).read_text(encoding="utf-8"))


def _normalize_manual_cell(
    cell: object,
) -> Tuple[Optional[int], str]:
    if cell is None:
        return None, "empty"
    if isinstance(cell, int):
        return int(cell), "confirmed_number"
    if isinstance(cell, str):
        label = cell.strip().lower()
        if label in ("black", "unknown", "empty"):
            return None, label
        raise ManualInputError("manual_grid_cell_label_invalid")
    if isinstance(cell, dict):
        label = str(cell.get("label", "")).strip().lower()
        if label == "number":
            if "value" not in cell:
                raise ManualInputError("manual_grid_number_missing_value")
            return int(cell["value"]), "confirmed_number"
        if label in ("black", "unknown", "empty"):
            return None, label
    raise ManualInputError("manual_grid_cell_invalid")


def _normalize_grid_payload(
    payload: object,
) -> tuple[List[List[Optional[int]]], List[List[str]]]:
    if isinstance(payload, dict):
        if "grid" not in payload:
            raise ManualInputError("manual_grid_missing_grid_key")
        payload = payload["grid"]
    if not isinstance(payload, list) or not payload:
        raise ManualInputError("manual_grid_invalid")
    out: List[List[Optional[int]]] = []
    states: List[List[str]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ManualInputError("manual_grid_invalid_row")
        out_row: List[Optional[int]] = []
        state_row: List[str] = []
        for v in row:
            value, state = _normalize_manual_cell(v)
            out_row.append(value)
            state_row.append(state)
        out.append(out_row)
        states.append(state_row)
    width = len(out[0])
    if any(len(r) != width for r in out):
        raise ManualInputError("manual_grid_non_rectangular")
    return out, states


def load_manual_grid(manual_grid_path: str | None) -> List[List[Optional[int]]] | None:
    payload = _read_json(manual_grid_path)
    if payload is None:
        return None
    grid, _ = _normalize_grid_payload(payload)
    return grid


def load_manual_grid_with_states(
    manual_grid_path: str | None,
) -> tuple[List[List[Optional[int]]] | None, List[List[str]] | None]:
    payload = _read_json(manual_grid_path)
    if payload is None:
        return None, None
    return _normalize_grid_payload(payload)


def apply_overrides(
    base_grid: List[List[Optional[int]]], override_path: str | None
) -> tuple[List[List[Optional[int]]], List[Dict[str, object]]]:
    if override_path is None:
        return [row[:] for row in base_grid], []
    payload = _read_json(override_path)
    if not isinstance(payload, list):
        raise ManualInputError("override_invalid")

    grid = [row[:] for row in base_grid]
    audit: List[Dict[str, object]] = []
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for idx, patch in enumerate(payload):
        if not isinstance(patch, dict):
            raise ManualInputError("override_item_invalid")
        if "row" not in patch or "col" not in patch or "label" not in patch:
            raise ManualInputError("override_item_missing_keys")
        r = int(patch["row"]) - 1
        c = int(patch["col"]) - 1
        if not (0 <= r < rows and 0 <= c < cols):
            raise ManualInputError("override_out_of_bounds")
        label = str(patch["label"])
        if label == "number":
            if "value" not in patch:
                raise ManualInputError("override_number_missing_value")
            grid[r][c] = int(patch["value"])
        elif label in ("black", "empty"):
            grid[r][c] = None
        else:
            raise ManualInputError("override_label_invalid")
        audit.append(
            {
                "index": idx,
                "row": r,
                "col": c,
                "label": label,
                "note": patch.get("note"),
            }
        )
    return grid, audit
