"""Utility functions for the LightGBM pipeline."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
