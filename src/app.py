"""FastAPI application exposing a health endpoint."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI()


@app.get("/ping")
def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "pong"}
