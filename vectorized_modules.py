
"""Stub of vectorized_modules – syntactically correct, no-op implementations."""
import numpy as np

def proximity_score(grid: np.ndarray) -> np.ndarray:
    """Return uniform score matrix (0.5)."""
    return np.full(grid.shape, 0.5, dtype=np.float32)

SCORING_MODULES = {
    'proximity_score': proximity_score,
}
