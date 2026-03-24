"""Azure Web App FastAPI entrypoint."""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException

from src.inference import predict

app = FastAPI(title="coco-ranking-api")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/predict")
def predict_api() -> Dict[str, object]:
    try:
        return predict()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
