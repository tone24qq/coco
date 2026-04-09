from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ParseSnapshot(BaseModel):
    raw_cells: List[Any] = Field(default_factory=list)
    notes: str = ""


class InferTargetPositionRequest(BaseModel):
    board: List[List[int]]
    target_number: int
    source: str = "gpt_image_parse"
    parse_snapshot: ParseSnapshot = Field(default_factory=ParseSnapshot)

    @field_validator("board")
    @classmethod
    def validate_board_non_empty(cls, value: List[List[int]]) -> List[List[int]]:
        if not value:
            raise ValueError("board must be a non-empty 2D array")
        if not all(isinstance(row, list) and row for row in value):
            raise ValueError("board rows must be non-empty arrays")
        width = len(value[0])
        if any(len(row) != width for row in value):
            raise ValueError("board must be rectangular")
        return value

    @model_validator(mode="after")
    def validate_cell_values(self) -> "InferTargetPositionRequest":
        for r_idx, row in enumerate(self.board):
            for c_idx, cell in enumerate(row):
                if not isinstance(cell, int):
                    raise ValueError(f"board[{r_idx}][{c_idx}] must be integer")
                if cell == 0 or cell < -1:
                    raise ValueError(f"board[{r_idx}][{c_idx}] must be -1 or positive integer")
        return self


class Cell(BaseModel):
    row: int
    col: int


class BestCell(Cell):
    score: float
    confidence_1_to_100: float


class CandidateCell(BestCell):
    module_scores: Dict[str, float]


class BoardShape(BaseModel):
    rows: int
    cols: int


class InferenceMetadata(BaseModel):
    score_type: str
    confidence_type: str
    confidence_1_to_100_type: str
    confidence_1_to_100_is_probability: bool
    source: str
    version: str


class InferTargetPositionResponse(BaseModel):
    status: Literal["ok", "already_opened"]
    board_shape: BoardShape
    target_number: int
    remaining_numbers: List[int]
    unopened_cells: List[Cell]
    best_cell: Optional[BestCell]
    candidate_cells: List[CandidateCell]
    confidence_score: float
    reasoning: List[str]
    module_contributions: Dict[str, float]
    metadata: InferenceMetadata

    model_config = ConfigDict(extra="forbid")
