# tests/conftest.py
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

def make_grid(r: int, c: int, hidden=(-1,)):
    """產生唯一數字且保留一格隱藏值 (-1) 的盤面。"""
    grid = np.arange(1, r*c+1, dtype=int).reshape(r, c)
    grid[(r//2), (c//2)] = hidden[0]
    return grid.tolist()