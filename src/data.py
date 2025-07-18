"""Data utilities for LightGBM training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_board_dataset(
    path: str | Path | None, *, n_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or generate a dataset of scratch-card boards.

    Parameters
    ----------
    path:
        Optional CSV file path. If ``None``, a random dataset is generated.
    n_samples:
        Number of samples to generate when ``path`` is ``None``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Boards (as 2D arrays) and corresponding labels.
    """
    if path is None:
        rng = np.random.default_rng(42)
        boards = rng.integers(0, 10, size=(n_samples, 5, 5))
        boards[rng.random(size=boards.shape) < 0.1] = -1
        labels = rng.integers(0, 2, size=n_samples)
        return boards, labels

    path = Path(path)
    df = pd.read_csv(path)
    board_cols = [c for c in df.columns if c.startswith("v")]
    boards = df[board_cols].to_numpy().reshape(len(df), 5, 5)
    labels = df["label"].to_numpy()
    return boards, labels
