# tests/conftest.py
import importlib.util
import pathlib
import sys
import warnings
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))  # noqa: E402
spec = importlib.util.spec_from_file_location("app", ROOT_DIR / "app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
app = module.app

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(all="ignore")


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Shared TestClient for FastAPI app."""

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def make_grid():
    """Return a helper to create an NxM grid with a hidden cell (-1)."""

    def _make(r: int, c: int, hidden=(-1,)):
        grid = np.arange(1, r * c + 1, dtype=int).reshape(r, c)
        grid[(r // 2), (c // 2)] = hidden[0]
        return grid.tolist()

    return _make
