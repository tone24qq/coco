from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class ManualInputError(ValueError):
    pass


def _read_json(path: str | None) -> object | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_grid_payload(payload: object) -> List[List[Optional[int]]]:
    if isinstance(payload, dict):
        if "grid" not in payload:
            raise ManualInputError("manual_grid_missing_grid_key")
        payload = payload["grid"]
    if not isinstance(payload, list) or not payload:
        raise ManualInputError("manual_grid_invalid")
    out: List[List[Optional[int]]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ManualInputError("manual_grid_invalid_row")
        out_row: List[Optional[int]] = []
        for v in row:
            if v is None:
                out_row.append(None)
            else:
                out_row.append(int(v))
        out.append(out_row)
    width = len(out[0])
    if any(len(r) != width for r in out):
        raise ManualInputError("manual_grid_non_rectangular")
    return out


def load_manual_grid(manual_grid_path: str | None) -> List[List[Optional[int]]] | None:
    payload = _read_json(manual_grid_path)
    if payload is None:
        return None
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
