import json
from pathlib import Path

import pytest

from src.manual_board_input import ManualInputError, apply_overrides, load_manual_grid


def test_load_full_manual_grid(tmp_path: Path) -> None:
    p = tmp_path / "grid.json"
    p.write_text(json.dumps({"grid": [[1, 2], [3, 4]]}), encoding="utf-8")
    grid = load_manual_grid(str(p))
    assert grid == [[1, 2], [3, 4]]


def test_apply_override_success(tmp_path: Path) -> None:
    p = tmp_path / "override.json"
    p.write_text(
        json.dumps([{"row": 2, "col": 2, "label": "number", "value": 9}]),
        encoding="utf-8",
    )
    grid, audit = apply_overrides([[1, 2], [3, None]], str(p))
    assert grid == [[1, 2], [3, 9]]
    assert audit[0]["label"] == "number"


def test_apply_override_out_of_bounds(tmp_path: Path) -> None:
    p = tmp_path / "override.json"
    p.write_text(
        json.dumps([{"row": 9, "col": 1, "label": "number", "value": 5}]),
        encoding="utf-8",
    )
    with pytest.raises(ManualInputError):
        apply_overrides([[1, 2], [3, 4]], str(p))


def test_load_manual_grid_supports_cell_labels(tmp_path: Path) -> None:
    p = tmp_path / "grid.json"
    p.write_text(
        json.dumps(
            {
                "grid": [
                    [1, "black"],
                    ["unknown", {"label": "number", "value": 4}],
                ]
            }
        ),
        encoding="utf-8",
    )
    grid = load_manual_grid(str(p))
    assert grid == [[1, None], [None, 4]]
