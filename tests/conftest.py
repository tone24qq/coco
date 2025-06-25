# tests/conftest.py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import app



@pytest.fixture(scope="session")
def client() -> TestClient:
    """Shared TestClient for FastAPI app."""

    return TestClient(app)


@pytest.fixture()
def make_grid():
    """Return a helper to create an NxM grid with a hidden cell (-1)."""

    def _make(r: int, c: int, hidden=(-1,)):
        grid = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
        grid[(r // 2), (c // 2)] = hidden[0]
        return grid.tolist()

    return _make
