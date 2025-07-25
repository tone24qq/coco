"""Entry point for running the FastAPI service."""

from api.main import app  # re-export

__all__ = ["app"]
