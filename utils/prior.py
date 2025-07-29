from __future__ import annotations

"""Utilities for heatmap priors."""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def build_heatmap(
    records: Iterable[Tuple[np.ndarray, Tuple[int, int]]], rows: int, cols: int
) -> np.ndarray:
    """Return Laplace-smoothed heatmap from ``records``."""
    mat = np.ones((rows, cols), dtype=np.float32)
    for board, pos in records:
        r, c = pos
        mat[r, c] += 1
    mat /= mat.sum()
    return mat


def load_heatmap(rows: int, cols: int, directory: str = "priors") -> "torch.Tensor":
    """Load heatmap ``rows`` x ``cols`` from ``directory``."""
    path = Path(directory) / f"heatmap_{rows}x{cols}.npy"
    arr = np.load(path)
    if arr.shape != (rows, cols):
        raise ValueError("heatmap shape mismatch")
    tensor = torch.from_numpy(arr.astype(np.float32)) if TORCH_AVAILABLE else arr
    return tensor
