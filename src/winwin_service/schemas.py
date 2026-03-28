from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    target_period: int
    latest_period: int
    top3: list[list[int]]
    kill_zone: list[int]
    metadata: dict[str, object] = Field(default_factory=dict)
