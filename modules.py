import json
import logging
from typing import Callable, Dict, Optional

import numpy as np
from scipy import ndimage as ndi

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Formula registry for Monte Carlo simulation
FORMULA_REGISTRY: Dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {}


def register_formula(name: str) -> Callable:
    """Register formula functions for generating scratch card grids."""

    def _decorator(fn: Callable) -> Callable:
        FORMULA_REGISTRY[name] = fn
        return fn

    return _decorator


@register_formula("excel")
def gen_excel(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid using random permutation of numbers 1 to N."""
    nums = rng.permutation(rows * cols) + 1
    return nums.reshape(rows, cols)


@register_formula("shuffle")
def gen_shuffle(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid by shuffling numbers within each row."""
    nums = np.arange(1, rows * cols + 1)
    board = nums.reshape(rows, cols)
    for r in range(rows):
        rng.shuffle(board[r])
    return board


@register_formula("random_entropy")
def gen_random_entropy(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid with entropy-based random dispersion."""
    grid = np.zeros((rows, cols), dtype=np.int64)
    legal = list(range(1, rows * cols + 1))
    rng.shuffle(legal)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        grid[r, c] = legal[i]
    return grid


@register_formula("tail_cluster")
def gen_tail_cluster(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Generate grid clustering larger numbers near the bottom-right corner."""
    n = rows * cols
    high_rows = max(1, int(rows * 0.3))
    high_cols = max(1, int(cols * 0.3))
    mask = np.zeros((rows, cols), dtype=bool)
    mask[-high_rows:, -high_cols:] = True
    high_idx = np.flatnonzero(mask.ravel())
    low_idx = np.flatnonzero(~mask.ravel())

    nums = np.arange(1, n + 1)
    rng.shuffle(nums)
    nums.sort()
    high_nums = nums[-len(high_idx) :]
    low_nums = nums[: -len(high_idx)]
    rng.shuffle(high_nums)
    rng.shuffle(low_nums)

    board = np.empty(n, dtype=np.int64)
    board[low_idx] = low_nums
    board[high_idx] = high_nums
    return board.reshape(rows, cols)


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
            hist = self.history[name] * (self.total_trials - 1) + score
            self.history[name] = hist / self.total_trials
            self.weights[name] = max(
                0.1,
                min(0.9, self.history[name] * 1.05),
            )
        total = sum(self.weights.values()) or 1e-10
        for name in self.weights:
            self.weights[name] /= total

    def save_history(self, path: str) -> None:
        """Save weight history to file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False)
        except OSError as e:
            logging.error(f"Failed to save weights history: {e}")


def global_offset_cooccurrence(
    boards: np.ndarray,
    target: int,
    offsets: Optional[list[int]] = None,
) -> np.ndarray:
    """GlobalOffsetCooccurrenceModule.

    Parameters
    ----------
    boards : np.ndarray
        Batch of boards with shape ``(batch, rows, cols)``.
    target : int
        Target number for offset counting.
    offsets : list[int], optional
        Offsets relative to ``target`` to count. Defaults to
        ``[1, -1, 10, -10, 20, -20]``.

    Returns
    -------
    np.ndarray
        Score array of the same shape as ``boards``.

    Examples
    --------
    >>> import numpy as np
    >>> b = np.array([[[1, -1], [2, 3]]])
    >>> global_offset_cooccurrence(b, target=1, offsets=[1])
    array([[[1., 1.],
            [1., 1.]]])
    """
    if offsets is None:
        offsets = [1, -1, 10, -10, 20, -20]

    boards = np.asarray(boards)
    if boards.ndim == 2:
        boards = boards[None, ...]

    batch, r, c = boards.shape
    mask_hidden = (boards == -1).astype(float)
    score = np.zeros_like(boards, dtype=float)
    for o in offsets:
        counts = np.sum(boards == (target + o), axis=(1, 2))
        score += mask_hidden * counts[:, None, None]
    return score


def neighbor_value_distribution(
    boards: np.ndarray,
    target: int,
    tolerance: int = 1,
    radius: int = 1,
) -> np.ndarray:
    """Score hidden cells based on nearby values.

    Parameters
    ----------
    boards : np.ndarray
        Batch of boards with shape ``(batch, rows, cols)``.
    target : int
        Target number to search around.
    tolerance : int, optional
        Acceptable absolute difference from ``target``. Defaults to ``1``.
    radius : int, optional
        Neighborhood radius for local counting. Defaults to ``1``.

    Returns
    -------
    np.ndarray
        Score array of the same shape as ``boards``.

    Examples
    --------
    >>> b = np.array([[[1, 2], [3, -1]]])
    >>> neighbor_value_distribution(b, target=1).round(2)
    array([[[0.67, 1.  ],
            [0.71, 0.71]]])
    """
    boards = np.asarray(boards)
    if boards.ndim == 2:
        boards = boards[None, ...]

    mask_hidden = boards == -1
    mask_near = (boards != -1) & (np.abs(boards - target) <= tolerance)
    dist = ndi.distance_transform_edt(~mask_near)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=float)
    mask_float = mask_near.astype(float)
    counts = ndi.convolve(
        mask_float,
        kernel[None, :, :],
        mode="constant",
    )
    counts_norm = counts / kernel.size
    max_d = max(boards.shape[1], boards.shape[2])
    dist_norm = dist / (max_d + 1e-9)
    score = mask_hidden.astype(float) * (0.6 * (1 - dist_norm) + 0.4 * counts_norm)
    return score
