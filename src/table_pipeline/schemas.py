from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CellRecord:
    row_index: int
    col_index: int
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float
    is_numeric: bool
    normalized_value: int | None
    review_needed: bool
    label: str
    top_candidates: list[dict[str, Any]]


@dataclass
class TableRecord:
    table_index: int
    board_bbox: tuple[int, int, int, int]
    rows: int
    cols: int
    cells: list[CellRecord]


@dataclass
class PipelineRecord:
    is_table_document: bool
    tables: list[TableRecord]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
