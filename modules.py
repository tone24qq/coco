from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ortools.sat.python import cp_model
from scipy import ndimage as ndi
from scipy.signal import convolve2d

from weights import BASE_WEIGHTS

# Formula registry for Monte Carlo simulation
FORMULA_REGISTRY: Dict[
    str,
    Callable[[int, int, np.random.Generator], np.ndarray],
] = {}


@dataclass
class ModuleStrategy:
    """Container for a scoring module."""

    func: Callable[[np.ndarray], np.ndarray]
    weight: float


STRATEGY_REGISTRY: Dict[str, ModuleStrategy] = {}


def register_strategy(name: str, *, weight: float) -> Callable[[Callable], Callable]:
    """Decorator to register a scoring strategy."""

    def _decorator(fn: Callable) -> Callable:
        STRATEGY_REGISTRY[name] = ModuleStrategy(fn, weight)
        return fn

    return _decorator


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


def generate_excel_style_card(
    rows: int, cols: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Return an ``rows x cols`` grid filled with unique numbers."""

    if rng is None:
        rng = np.random.default_rng()
    n = rows * cols
    values = rng.choice(np.arange(1, n + 1), size=n, replace=False)
    return values.reshape((rows, cols))


def sample_similarity(
    partial_grid: List[List[int]],
    full_sample: List[List[int]],
) -> float:
    """Return similarity ratio between ``partial_grid`` and ``full_sample``."""

    total = 0
    match = 0
    rows = len(partial_grid)
    cols = len(partial_grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            v = partial_grid[r][c]
            if v != -1:
                total += 1
                if full_sample[r][c] == v:
                    match += 1
    return match / total if total > 0 else 0.0


def locate_target_by_partial_grid(
    grid: List[List[int]],
    target: int,
    *,
    sample_library: Optional[List[List[List[int]]]] = None,
    use_legacy: bool = False,
) -> Tuple[int, int]:
    """Predict the target's location from a partially hidden grid.

    The default implementation uses a CSP solver to infer the most likely
    position of ``target`` when it is hidden. The previous random fallback
    logic is still available via ``use_legacy=True``.
    """

    arr = np.asarray(grid, dtype=int)
    if arr.ndim != 2:
        raise ValueError("grid must be a 2D array")

    idx = np.argwhere(arr == target)
    if idx.size:
        r, c = idx[0]
        return int(r), int(c)

    hidden = np.argwhere(arr == -1)
    if hidden.size == 0:
        raise ValueError("target not found and no hidden cells available")

    blanks = [tuple(p) for p in hidden]

    if use_legacy:
        rng = np.random.default_rng()
        r, c = blanks[int(rng.integers(len(blanks)))]
        return int(r), int(c)

    if sample_library:
        for sample in sample_library:
            if sample_similarity(grid, sample) >= 0.5:
                arr_s = np.asarray(sample)
                pos = np.argwhere(arr_s == target)
                if pos.size:
                    r, c = pos[0]
                    if arr[r, c] == -1:
                        return int(r), int(c)
                break

    missing_numbers = [n for n in range(1, arr.size + 1) if n not in arr[arr != -1]]

    model = cp_model.CpModel()
    vars_x = [
        model.NewIntVarFromDomain(cp_model.Domain.FromValues(missing_numbers), f"x{i}")
        for i in range(len(blanks))
    ]
    model.AddAllDifferent(vars_x)
    bools = []
    for var in vars_x:
        b = model.NewBoolVar("")
        model.Add(var == target).OnlyEnforceIf(b)
        model.Add(var != target).OnlyEnforceIf(b.Not())
        bools.append(b)
    model.Add(sum(bools) == 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.02

    class Counter(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables):
            super().__init__()
            self.variables = variables
            self.count = 0
            self.freq = {pos: 0 for pos in blanks}

        def on_solution_callback(self):
            self.count += 1
            for var, pos in zip(self.variables, blanks):
                if self.Value(var) == target:
                    self.freq[pos] += 1
            if self.count >= 16:
                self.StopSearch()

    cb = Counter(vars_x)
    solver.SearchForAllSolutions(model, cb)

    if cb.count == 0:
        rng = np.random.default_rng()
        r, c = blanks[int(rng.integers(len(blanks)))]
        return int(r), int(c)

    best = max(cb.freq.items(), key=lambda kv: kv[1])[0]
    return int(best[0]), int(best[1])


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
    nearest_k: int = 0,
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
    # 回退：若找不到任意符合 tolerance 的位置，則使用最近的 `nearest_k` 個值
    if not mask_near.any() and nearest_k > 0:
        for i, board in enumerate(boards):
            if (np.abs(board - target) <= tolerance).any():
                continue
            known = np.unique(board[board != -1])
            if known.size:
                idx = np.argsort(np.abs(known - target))[:nearest_k]
                near_k = known[idx]
                mask_near[i] = np.isin(board, near_k)
            else:
                mask_near[i] = False
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


def nearest_value_affinity(
    grid: np.ndarray,
    target: Optional[int],
    *,
    k: int = 3,
    tolerance: int = 1,
    radius: int = 1,
) -> np.ndarray:
    """Return normalized affinity heatmap using :func:`neighbor_value_distribution`.

    Parameters
    ----------
    grid : np.ndarray
        Board matrix with ``-1`` for unknown cells.
    target : int
        Target number to search around.
    k : int, optional
        Number of nearest values to fall back to when no cells match ``tolerance``.
    tolerance : int, optional
        Acceptable absolute difference from ``target``. Defaults to ``1``.
    radius : int, optional
        Neighborhood radius for local counting. Defaults to ``1``.
    """

    score = neighbor_value_distribution(
        grid,
        target=target,
        tolerance=tolerance,
        radius=radius,
        nearest_k=k,
    )
    mx = score.max(initial=0.0)
    if mx > 0:
        score = score / float(mx)
    return score


@register_strategy("focus", weight=0.2)
def compute_focus_score(grid: np.ndarray) -> np.ndarray:
    """Compute density of known cells in 3x3 neighborhood."""
    mask = (grid != -1).astype(int)
    kernel = np.ones((3, 3), dtype=int)
    raw = convolve2d(mask, kernel, mode="same", boundary="fill", fillvalue=0)
    result = np.zeros_like(raw, dtype=float)
    result[grid == -1] = raw[grid == -1]
    mn, mx = result.min(), result.max()
    if mx > mn:
        result = (result - mn) / (mx - mn)
    return result


@register_strategy("skip", weight=0.15)
def detect_skip_patterns(grid: np.ndarray) -> np.ndarray:
    """Detect arithmetic skip patterns along rows and columns."""
    M, N = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(M):
        idx = np.where(grid[i] != -1)[0]
        if idx.size < 2:
            continue
        start, end = idx[0], idx[-1]
        for j in range(start, N):
            if grid[i, j] == -1:
                score[i, j] = 1
        for j in range(end, -1, -1):
            if grid[i, j] == -1:
                score[i, j] = 1
    for j in range(N):
        idx = np.where(grid[:, j] != -1)[0]
        if idx.size < 2:
            continue
        start, end = idx[0], idx[-1]
        for i2 in range(start, M):
            if grid[i2, j] == -1:
                score[i2, j] = 1
        for i2 in range(end, -1, -1):
            if grid[i2, j] == -1:
                score[i2, j] = 1
    return score


@register_strategy("diff", weight=0.15)
def compute_difference_trend(grid: np.ndarray) -> np.ndarray:
    """Infer values based on local difference trend."""
    M, N = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(M):
        for j in range(N):
            if grid[i, j] != -1:
                continue
            if j >= 2 and grid[i, j - 1] != -1 and grid[i, j - 2] != -1:
                score[i, j] = max(score[i, j], 1)
            if i >= 2 and grid[i - 1, j] != -1 and grid[i - 2, j] != -1:
                score[i, j] = max(score[i, j], 1)
    return score


@register_strategy("mirror", weight=0.2)
def detect_mirror_sequences(grid: np.ndarray) -> np.ndarray:
    """Check horizontal and vertical mirror symmetry."""
    M, N = grid.shape
    score = np.zeros_like(grid, dtype=float)
    for i in range(M):
        for j in range(N):
            if grid[i, j] != -1:
                continue
            max_k = min(j, N - j - 1)
            for k in range(1, max_k + 1):
                left, right = grid[i, j - k], grid[i, j + k]
                if left != -1 and right != -1 and left == right:
                    score[i, j] = 1
                    break
            if score[i, j] == 1:
                continue
            max_k2 = min(i, M - i - 1)
            for k in range(1, max_k2 + 1):
                up, down = grid[i - k, j], grid[i + k, j]
                if up != -1 and down != -1 and up == down:
                    score[i, j] = 1
                    break
    return score


@register_strategy("conn", weight=0.15)
def connectivity_heatmap(grid: np.ndarray) -> np.ndarray:
    """Distance weighted sum of known numbers."""
    M, N = grid.shape
    score = np.zeros_like(grid, dtype=float)
    known = np.argwhere(grid != -1)
    if known.size == 0:
        return score
    for i in range(M):
        for j in range(N):
            if grid[i, j] != -1:
                continue
            d = known - np.array([i, j])
            dist = np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2))
            score[i, j] = np.sum(1.0 / (dist + 1e-6))
    return score


@register_strategy("tail", weight=0.15)
def sequence_tail_analyzer(grid: np.ndarray) -> np.ndarray:
    """Analyze digit tails frequency weighted by distance."""
    M, N = grid.shape
    score = np.zeros_like(grid, dtype=float)
    known = np.argwhere(grid != -1)
    if known.size == 0:
        return score
    tail_pos = {t: [] for t in range(10)}
    for r, c in known:
        t = int(grid[r, c] % 10)
        tail_pos[t].append((r, c))
    tail_counts = {t: len(tail_pos[t]) for t in range(10)}
    for i in range(M):
        for j in range(N):
            if grid[i, j] != -1:
                continue
            best = 0.0
            for t in range(10):
                pos = tail_pos[t]
                if not pos:
                    continue
                coords = np.array(pos)
                dists = np.abs(coords - np.array([i, j])).sum(axis=1)
                min_d = np.min(dists)
                s = tail_counts[t] / (min_d + 1e-6)
                if s > best:
                    best = s
            score[i, j] = best
    return score


@register_strategy("affinity", weight=0.15)
def target_affinity(grid: np.ndarray, *, target: Optional[int] = None) -> np.ndarray:
    """Affinity based on nearest value distribution."""

    if target is None:
        return np.zeros_like(grid, dtype=float)
    return nearest_value_affinity(grid, target, k=3, tolerance=1, radius=1)


@register_strategy("gradient_affinity", weight=0.1)
def gradient_affinity(grid: np.ndarray) -> np.ndarray:
    """Score blanks based on local gradient continuity."""

    arr = np.asarray(grid, dtype=float)
    rows, cols = arr.shape
    score = np.zeros_like(arr, dtype=float)
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] != -1:
                continue
            vals = []
            if 0 < j < cols - 1 and arr[i, j - 1] != -1 and arr[i, j + 1] != -1:
                vals.append(abs(arr[i, j + 1] - arr[i, j - 1]))
            if 0 < i < rows - 1 and arr[i - 1, j] != -1 and arr[i + 1, j] != -1:
                vals.append(abs(arr[i + 1, j] - arr[i - 1, j]))
            if vals:
                score[i, j] = sum(vals) / len(vals)
    if score.max() > 0:
        score /= float(score.max())
    return score


@register_strategy("row_col_bias", weight=0.1)
def row_col_bias(grid: np.ndarray) -> np.ndarray:
    """Bias hidden cells by row/column known counts."""

    arr = np.asarray(grid)
    mask = arr != -1
    row_ratio = mask.sum(axis=1) / arr.shape[1]
    col_ratio = mask.sum(axis=0) / arr.shape[0]
    base = (1.0 - row_ratio)[:, None] + (1.0 - col_ratio)[None, :]
    base[arr != -1] = 0.0
    mn, mx = base.min(), base.max()
    if mx > mn:
        base = (base - mn) / (mx - mn)
    return base.astype(float)


@register_strategy("row_col_frequency_score", weight=0.1)
def row_col_frequency_score(
    grid: np.ndarray, *, target: Optional[int] = None
) -> np.ndarray:
    """Score cells using target proximity frequency in rows and columns."""

    if target is None:
        return np.zeros_like(grid, dtype=float)
    arr = np.asarray(grid, dtype=float)
    diff = np.abs(arr - float(target))
    diff[arr == -1] = np.nan
    row_score = np.nanmean(1.0 / (1.0 + diff), axis=1)
    col_score = np.nanmean(1.0 / (1.0 + diff), axis=0)
    score = row_score[:, None] + col_score[None, :]
    score[arr != -1] = 0.0
    mn, mx = np.nanmin(score), np.nanmax(score)
    if mx > mn:
        score = (score - mn) / (mx - mn)
    return np.nan_to_num(score, nan=0.0)


@register_strategy("entropy_spread_score", weight=0.1)
def entropy_spread_score(grid: np.ndarray) -> np.ndarray:
    """Shannon entropy of neighbor digit tails."""

    arr = np.asarray(grid)
    rows, cols = arr.shape
    score = np.zeros_like(arr, dtype=float)
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] != -1:
                continue
            r0, r1 = max(0, i - 1), min(rows, i + 2)
            c0, c1 = max(0, j - 1), min(cols, j + 2)
            neigh = arr[r0:r1, c0:c1]
            vals = neigh[neigh != -1] % 10
            if vals.size:
                _, counts = np.unique(vals, return_counts=True)
                probs = counts / counts.sum()
                ent = -(probs * np.log2(probs)).sum()
                score[i, j] = ent
    if score.max() > 0:
        score /= float(score.max())
    return score


def fuse_scores(
    gridscores: Dict[str, np.ndarray],
    grid: np.ndarray,
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    """Fuse score arrays from multiple modules."""
    if weights is None:
        weights = BASE_WEIGHTS
    final = np.zeros_like(grid, dtype=float)
    for name, arr in gridscores.items():
        final += weights.get(name, 0.0) * arr
    final[grid != -1] = 0
    mn, mx = final.min(), final.max()
    if mx > mn:
        final = (final - mn) / (mx - mn)
    return final


# Load vectorized implementations which override default strategies
try:  # noqa: WPS501
    import advanced_patterns_vect as _apv  # noqa: F401

    compute_focus_score = _apv.compute_focus_score  # noqa: F811
    detect_skip_patterns = _apv.detect_skip_patterns  # noqa: F811
    compute_difference_trend = _apv.compute_difference_trend  # noqa: F811
    detect_mirror_sequences = _apv.detect_mirror_sequences  # noqa: F811
    connectivity_heatmap = _apv.connectivity_heatmap  # noqa: F811
    sequence_tail_analyzer = _apv.sequence_tail_analyzer  # noqa: F811
    diagonal_consistency_score = _apv.diagonal_consistency_score  # noqa: F811
    row_col_bias = _apv.row_col_bias  # noqa: F811
    fuse_scores_vect = _apv.fuse_scores_vect
except Exception:  # pragma: no cover - optional dependency
    pass

# ------------------------------------------------------------------
# 新模組：全域 1~N 等差分析 (full_range_arith_score)
# ------------------------------------------------------------------
from math import sqrt  # noqa: E402


def _divisors(n: int) -> list[int]:
    out = []
    for i in range(1, int(sqrt(n)) + 1):
        if n % i == 0:
            out.extend([i, n // i])
    return sorted(set(out))


def full_range_arith_score(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=int)
    rows, cols = arr.shape
    N = rows * cols

    filled = arr[arr != -1]
    if filled.size < 3:
        return np.zeros_like(arr, dtype=float)

    covers = []
    divs = _divisors(N - 1)
    for d in divs:
        covers.append(((filled - 1) % d == 0).mean())
    d_star = divs[int(np.argmax(covers))]
    if max(covers) < 0.3:
        return np.zeros_like(arr, dtype=float)

    missing = set(range(1, N + 1)) - set(filled)
    missing_in_prog = {x for x in missing if (x - 1) % d_star == 0}
    if not missing_in_prog:
        return np.zeros_like(arr, dtype=float)

    score = np.zeros_like(arr, dtype=float)
    score[arr == -1] = len(missing_in_prog) / len(missing)
    return score


register_strategy("gdiff", weight=0.0)(full_range_arith_score)

# 讀取集中式權重
try:  # noqa: WPS501
    from weights_config import WEIGHTS as USER_WEIGHTS

    BASE_WEIGHTS.update(USER_WEIGHTS)
except Exception:
    pass
