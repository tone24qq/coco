import numpy as np

from src.table_pipeline.global_decode import decode_with_constraints
from src.table_pipeline.table_detect import _line_based_grid


def test_detected_lines_not_even_split_override() -> None:
    h, w = 300, 240
    img = np.zeros((h, w), dtype=np.uint8)
    row_lines = [0, 52, 113, 171, 239, 299]
    col_lines = [0, 57, 121, 178, 239]
    for y in row_lines:
        img[max(0, y - 1) : min(h, y + 1), :] = 255
    for x in col_lines:
        img[:, max(0, x - 1) : min(w, x + 1)] = 255

    rows, cols, _diag = _line_based_grid(img)
    assert len(rows) == 6
    assert len(cols) == 5
    # not replaced by pure np.linspace average split
    assert rows != list(np.linspace(0, h - 1, 6, dtype=int))
    assert cols != list(np.linspace(0, w - 1, 5, dtype=int))


def test_global_decode_removes_duplicates_and_illegal() -> None:
    cands = {
        (0, 0): [{"value": 1, "score": 0.9}, {"value": 2, "score": 0.5}],
        (0, 1): [{"value": 1, "score": 0.88}, {"value": 2, "score": 0.87}],
        (1, 0): [{"value": 3, "score": 0.85}],
        (1, 1): [{"value": 4, "score": 0.84}, {"value": 9, "score": 0.99}],
    }
    decoded = decode_with_constraints(2, 2, cands)
    nums = [v for row in decoded.grid for v in row if v is not None]
    assert sorted(nums) == [1, 2, 3, 4]


def test_80_120_not_template_only_backend() -> None:
    from src.cell_digit_reader import read_cell_digit

    cell = np.full((64, 64), 255, dtype=np.uint8)
    out = read_cell_digit(cell, max_value=120)
    assert out.ocr_backend != "fallback_template"
