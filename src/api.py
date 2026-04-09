from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.inference_facade import infer_target_position
from src.inference_models import InferTargetPositionRequest, InferTargetPositionResponse
from src.inference_service import InferenceError

app = FastAPI(title="Scratchcard Board Inference Service", version="v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer_target_position", response_model=InferTargetPositionResponse)
def infer_target_position_api(payload: InferTargetPositionRequest) -> InferTargetPositionResponse:
    try:
        result = infer_target_position(
            board=payload.board,
            target_number=payload.target_number,
            source=payload.source,
        )
        return InferTargetPositionResponse(**result)
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
