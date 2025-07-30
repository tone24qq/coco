from __future__ import annotations

"""Utilities for heatmap priors."""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from dataset import BLANK_VALUE

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


def bucket_of(val: int) -> str:
    """Return ``"small"`` | ``"mid"`` | ``"large"`` for ``val``."""

    return "small" if val <= 5 else "mid" if val <= 15 else "large"


def load_heatmap(
    rows: int,
    cols: int,
    target: int | None = None,
    device: str | "torch.device" = "cpu",
) -> "torch.Tensor":
    """Load a smoothed heatmap prior.

    Parameters
    ----------
    rows, cols
        Board shape.
    target
        Target value.  ``None`` uses the legacy single-map path.
    device
        Torch device string or object for output tensor.
    """

    if target is None:
        fname = f"heatmap_{rows}x{cols}.npy"
    else:
        fname = f"heatmap_{bucket_of(target)}_{rows}x{cols}.npy"

    path = Path("priors") / fname
    if not path.exists():  # graceful fallback to uniform prior
        if torch is not None:
            return torch.full((rows, cols), 1.0 / (rows * cols), device=device)
        return np.full((rows, cols), 1.0 / (rows * cols), dtype=np.float32)

    arr = np.load(path)
    print(f"Loaded heatmap: {fname}")
    if arr.shape != (rows, cols):
        raise ValueError("heatmap shape mismatch")
    if torch is not None:
        tensor = torch.tensor(arr, device=device, dtype=torch.float32)
    else:  # pragma: no cover - torch missing
        tensor = arr
    return tensor


def fuse_predictions_with_heatmap(
    model_scores: np.ndarray,
    heatmap: np.ndarray,
    board: np.ndarray,
    model_weight: float = 0.8,
    heatmap_weight: float = 0.2,
) -> np.ndarray:
    """Fuse ``model_scores`` with ``heatmap`` respecting masked cells."""

    if heatmap.shape != board.shape:
        raise ValueError("heatmap shape mismatch")
    if model_scores.ndim != 1 or model_scores.size != board.size:
        raise ValueError("model_scores shape mismatch")
    mask = (board.ravel() == BLANK_VALUE).astype(np.float32)
    prior = heatmap.ravel() * mask
    fused = model_weight * model_scores + heatmap_weight * prior
    return fused
