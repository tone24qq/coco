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


class Cell(BaseModel):
    row: int
    col: int


class BestCell(Cell):
    score: float
    confidence_1_to_100: float


class CandidateCell(BestCell):
    module_scores: Dict[str, float]
    module_details: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    support_score: float = 0.0
    contradiction_penalty: float = 0.0
    gated_score: float = 0.0
    ranking_score: float = 0.0
    final_score: float = 0.0
    gate_multiplier: float = 1.0


class BoardShape(BaseModel):
    rows: int
    cols: int


class InferenceMetadata(BaseModel):
    score_type: str
    score_can_be_negative: Optional[bool] = None
    confidence_score_is_not_ranking_score: Optional[bool] = None
    confidence_type: str
    confidence_1_to_100_type: str
    confidence_1_to_100_is_probability: bool
    best_cell_confidence_1_to_100: Optional[float] = None
    margin_to_top2: Optional[float] = None
    effective_candidate_count: Optional[int] = None
    gated_candidate_count: Optional[int] = None
    confidence_reason: Optional[str] = None
    raw_score_min: Optional[float] = None
    raw_score_max: Optional[float] = None
    raw_score_std: Optional[float] = None
    final_score_min: Optional[float] = None
    final_score_max: Optional[float] = None
    final_score_std: Optional[float] = None
    top1_top2_margin: Optional[float] = None
    top1_top5_mean_gap: Optional[float] = None
    score_entropy_like: Optional[float] = None
    collapsed_score_flag: Optional[bool] = None
    source: str
    version: str
    aggregation_type: Optional[str] = None
    normalization_mode: Optional[str] = None
    gating_enabled: Optional[bool] = None
    elimination_version: Optional[str] = None
    ranking_stage: Literal["baseline_only", "reranker_applied"]
    reranker_version: Optional[str]
    reranker_feature_schema_version: Optional[str]
    reranker_fallback_reason: Optional[str]


class InferTargetPositionResponse(BaseModel):
    status: Literal["ok", "already_opened"]
    board_shape: BoardShape
    target_number: int
    remaining_numbers: List[int]
    unopened_cells: List[Cell]
    best_cell: Optional[BestCell]
    candidate_cells: List[CandidateCell]
    confidence_score: float
    best_ranking_score: Optional[float] = None
    best_confidence_score: Optional[float] = None
    reasoning: List[str]
    module_contributions: Dict[str, float]
    metadata: InferenceMetadata

    model_config = ConfigDict(extra="forbid")


class MultiTargetAssignment(BaseModel):
    target_number: int
    row: int
    col: int
    joint_score: float
    base_score: float
    was_reassigned_from_individual_top1: bool
    individual_top1_row: int
    individual_top1_col: int
    reassignment_cost_delta: float


class InferMultiTargetPositionResponse(BaseModel):
    status: Literal["ok"]
    board_shape: BoardShape
    target_numbers: List[int]
    assignments: List[MultiTargetAssignment]
    assignment_score_table: Dict[str, Dict[str, float]]
    per_target_ranked_candidates: Dict[str, List[CandidateCell]]
    metadata: Dict[str, Any]
