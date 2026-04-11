from __future__ import annotations

from typing import Any, List

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


class InferMultiTargetPositionRequest(BaseModel):
    board: List[List[int]]
    target_numbers: List[int]
    source: str = "gpt_image_parse"
    parse_snapshot: ParseSnapshot = Field(default_factory=ParseSnapshot)

    @field_validator("board")
    @classmethod
    def validate_board_non_empty(cls, value: List[List[int]]) -> List[List[int]]:
        return InferTargetPositionRequest.validate_board_non_empty(value)

    @field_validator("target_numbers")
    @classmethod
    def validate_target_numbers(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("target_numbers must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("target_numbers must be unique")
        if any((not isinstance(v, int)) for v in value):
            raise ValueError("target_numbers must be integers")
        return value


class Top10Cell(BaseModel):
    row: int
    col: int
    confidence_1_to_100: float


class CompactInferenceResponse(BaseModel):
    top10: List[Top10Cell]
    best_confidence_1_to_100: float

    model_config = ConfigDict(extra="forbid")


InferTargetPositionResponse = CompactInferenceResponse
InferMultiTargetPositionResponse = CompactInferenceResponse
