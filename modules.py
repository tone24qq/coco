from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

# Formula registry for Monte Carlo simulation
FORMULA_REGISTRY: Dict[
    str,
    Callable[[int, int, np.random.Generator], np.ndarray],
] = {}


def generate_unique_grid(
    rows: int,
    cols: int,
    *,
    hidden: Union[int, Tuple[int, int], List[Tuple[int, int]], None] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a ``rows x cols`` grid with unique numbers and optional hidden cells.

    Parameters
    ----------
    rows, cols:
        Dimensions of the grid. Each must be between 4 and 20 inclusive.
    hidden:
        If ``None`` a random cell is hidden. If an ``int`` ``n`` is provided,
        ``n`` random cells are hidden. If a tuple ``(r, c)`` or a list of such
        tuples is given, those coordinates are hidden.
    rng:
        Optional random generator used for sampling positions.

    Returns
    -------
    np.ndarray
        Generated grid with ``-1`` in hidden positions.

    Raises
    ------
    ValueError
        If ``rows`` or ``cols`` is outside the allowed range.
    """

    if not 4 <= rows <= 20 or not 4 <= cols <= 20:
        raise ValueError("Grid size must be 4–20")

    if rng is None:
        rng = np.random.default_rng()

    grid = np.arange(1, rows * cols + 1, dtype=int).reshape(rows, cols)

    positions: List[Tuple[int, int]]
    if hidden is None:
        positions = [(int(rng.integers(rows)), int(rng.integers(cols)))]
    elif isinstance(hidden, int):
        if hidden < 1 or hidden > rows * cols:
            raise ValueError("Invalid hidden count")
        all_idx = [(r, c) for r in range(rows) for c in range(cols)]
        chosen = rng.choice(len(all_idx), hidden, replace=False)
        positions = [all_idx[i] for i in np.atleast_1d(chosen)]
    elif isinstance(hidden, tuple):
        positions = [hidden]
    else:
        positions = list(hidden)

    if positions:
        rr, cc = np.array(positions).T
        grid[rr.astype(int), cc.astype(int)] = -1
    return grid


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
def gen_random_entropy(
    rows: int, cols: int, rng: np.random.Generator
) -> np.ndarray:  # noqa: E501
    """Generate grid with entropy-based random dispersion."""
    nums = rng.permutation(rows * cols) + 1
    return nums.reshape(rows, cols)


@register_formula("tail_cluster")
def gen_tail_cluster(
    rows: int, cols: int, rng: np.random.Generator
) -> np.ndarray:  # noqa: E501
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
    idx = -len(high_idx)
    high_nums = nums[idx:]
    low_nums = nums[:idx]
    rng.shuffle(high_nums)
    rng.shuffle(low_nums)

    board = np.empty(n, dtype=np.int64)
    board[low_idx] = low_nums
    board[high_idx] = high_nums
    return board.reshape(rows, cols)


def global_offset_cooccurrence(
    boards: np.ndarray,
    target: Optional[int] = None,
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
    if target is None:
        return np.zeros_like(boards, dtype=float)

    if offsets is None:
        offsets = [1, -1, 10, -10, 20, -20]

    boards = np.asarray(boards)
    single = boards.ndim == 2
    if single:
        boards = boards[None, ...]

    batch, r, c = boards.shape
    mask_hidden = (boards == -1).astype(float)
    off = np.asarray(offsets, dtype=int)
    matches = boards[..., None] == (target + off)[None, None, None, :]
    counts = matches.sum(axis=(1, 2))  # (batch, len(offsets))
    score = mask_hidden * counts.sum(axis=1)[:, None, None]

    return score[0] if single else score


def neighbor_value_distribution(
    boards: np.ndarray,
    target: Optional[int] = None,
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
    single = boards.ndim == 2
    if single:
        boards = boards[None, ...]

    if target is None:
        zeros = np.zeros_like(boards)
        return zeros if not single else zeros[0]

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
    base = 0.6 * (1 - dist_norm) + 0.4 * counts_norm
    score = mask_hidden.astype(float) * base
    return score[0] if single else score
