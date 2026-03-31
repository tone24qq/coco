from __future__ import annotations

from pydantic import BaseModel, Field


class RankedTriplet(BaseModel):
    rank: int
    numbers: list[int]
    score: float
    confidence: float
    overlap_count_vs_previous: int
    high_confidence_overlap: bool


class PredictionResponse(BaseModel):
    target_period: int
    latest_period: int
    top3: list[list[int]]
    top10: list[RankedTriplet]
    top10_display: list[str]
    kill_zone: list[int]
    metadata: dict[str, object] = Field(default_factory=dict)
