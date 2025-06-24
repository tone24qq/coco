import numpy as np
from typing import Dict, Callable, Optional
import logging
import json

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Formula registry for Monte Carlo simulation
FORMULA_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {}

def register_formula(name: str) -> Callable:
    """Decorator to register formula functions for generating scratch card grids."""
    def _decorator(fn: Callable) -> Callable:
        FORMULA_REGISTRY[name] = fn
        return fn
    return _decorator

@register_formula("excel")
def gen_excel(rows: int, cols: int, rng: np.random.Generator, batch: int = 1) -> np.ndarray:
    """Generate batch of grids using random permutation of numbers 1..N."""
    base = np.arange(1, rows * cols + 1)
    rand = rng.random((batch, rows * cols))
    idx = np.argsort(rand, axis=1)
    boards = base[idx].reshape(batch, rows, cols)
    return boards.astype(np.int16)

@register_formula("shuffle")
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator, batch: int = 1) -> np.ndarray:
    """Generate batch of grids by shuffling numbers within each row."""
    base = np.arange(1, rows * cols + 1).reshape(rows, cols)
    base = np.broadcast_to(base, (batch, rows, cols)).copy()
    rand = rng.random((batch, rows, cols))
    idx = np.argsort(rand, axis=2)
    boards = np.take_along_axis(base, idx, axis=2)
    return boards.astype(np.int16)

@register_formula("random_entropy")
def gen_random_entropy(rows: int, cols: int, rng: np.random.Generator, batch: int = 1) -> np.ndarray:
    """Generate batch of grids with entropy-based random dispersion."""
    base = np.arange(1, rows * cols + 1)
    rand = rng.random((batch, rows * cols))
    idx = np.argsort(rand, axis=1)
    boards = base[idx].reshape(batch, rows, cols)
    return boards.astype(np.int16)

@register_formula("tail_cluster")
def gen_tail_cluster(rows: int, cols: int, rng: np.random.Generator, batch: int = 1) -> np.ndarray:
    """Generate batch of grids clustering larger numbers near the bottom-right corner."""
    n = rows * cols
    high_rows = max(1, int(rows * 0.3))
    high_cols = max(1, int(cols * 0.3))
    mask = np.zeros((rows, cols), dtype=bool)
    mask[-high_rows:, -high_cols:] = True
    high_idx = np.flatnonzero(mask.ravel())
    low_idx = np.flatnonzero(~mask.ravel())

    nums = np.arange(1, n + 1)
    rand = rng.random((batch, n))
    perm = nums[np.argsort(rand, axis=1)]
    perm.sort(axis=1)
    high_nums = perm[:, -len(high_idx):]
    low_nums = perm[:, :-len(high_idx)]
    boards = np.empty((batch, n), dtype=np.int16)
    boards[:, low_idx] = low_nums
    boards[:, high_idx] = high_nums
    return boards.reshape(batch, rows, cols)

class AdaptiveWeights:
    """Manages dynamic weight adjustments for formulas based on performance."""
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights.copy()
        self.history: Dict[str, float] = {name: 0.0 for name in initial_weights}
        self.total_trials = 0

    def update(self, success_rate: float, module_scores: Dict[str, float]) -> None:
        """Update weights based on success rate and module scores."""
        self.total_trials += 1
        for name in self.weights:
            score = module_scores.get(name, success_rate)
            self.history[name] = (self.history[name] * (self.total_trials - 1) + score) / self.total_trials
            self.weights[name] = max(0.1, min(0.9, self.history[name] * 1.05))
        total = sum(self.weights.values()) or 1e-10
        for name in self.weights:
            self.weights[name] /= total

    def save_history(self, path: str) -> None:
        """Save weight history to file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False)
        except OSError as e:
            logging.error(f"Failed to save weights history: {e}")

def global_offset_cooccurrence(
    boards: np.ndarray,
    target: int,
    offsets: Optional[list[int]] = None,
) -> np.ndarray:
    """GlobalOffsetCooccurrenceModule

    Count occurrences of ``target + offset`` across each board and assign the
    count to every hidden cell as a base score.

    Parameters
    ----------
    boards : np.ndarray
        Batch of boards with shape ``(batch, rows, cols)``.
    target : int
        Target number to offset from.
    offsets : list[int], optional
        Offsets to check; defaults to ``[1, -1, 10, -10, 20, -20]``.

    Returns
    -------
    np.ndarray
        Score array of shape ``(batch, rows, cols)``.
    """
    if offsets is None:
        offsets = [1, -1, 10, -10, 20, -20]

    boards = np.asarray(boards)
    if boards.ndim != 3:
        raise ValueError("boards must have shape (batch, rows, cols)")

    batch, r, c = boards.shape
    mask_hidden = (boards == -1).astype(float)
    score = np.zeros((batch, r, c), dtype=float)

    for o in offsets:
        counts = np.sum(boards == (target + o), axis=(1, 2))
        score += mask_hidden * counts[:, None, None]

    return score

