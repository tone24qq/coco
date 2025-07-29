from __future__ import annotations

"""Utilities for heatmap priors."""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def bucket_of(val: int) -> str:
    """Return bucket name for ``val``."""
    return "small" if val <= 5 else "mid" if val <= 15 else "large"


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


def load_heatmap(
    rows: int,
    cols: int,
    target: int | None = None,
    directory: str = "priors",
    *,
    device: str | None = None,
) -> "torch.Tensor":
    """Load heatmap for ``rows`` x ``cols`` optionally using ``target`` bucket."""

    if target is None:
        fname = f"heatmap_{rows}x{cols}.npy"
    else:
        fname = f"heatmap_{bucket_of(target)}_{rows}x{cols}.npy"
    path = Path(directory) / fname

    if not path.exists():
        if TORCH_AVAILABLE:
            import torch  # local

            return torch.full(
                (rows, cols),
                1.0 / (rows * cols),
                device=device,
                dtype=torch.float32,
            )
        return np.full((rows, cols), 1.0 / (rows * cols), dtype=np.float32)

    arr = np.load(path)
    if arr.shape != (rows, cols):
        raise ValueError("heatmap shape mismatch")
    if TORCH_AVAILABLE:
        import torch

        tensor = torch.tensor(arr, device=device, dtype=torch.float32)
    else:
        tensor = arr
    return tensor
