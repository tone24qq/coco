import inspect
import logging
import math
import os
import random
from collections import Counter, defaultdict
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import irfftn, rfftn
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2

from modules import global_offset_cooccurrence, neighbor_value_distribution

priors: Dict[int, float] = {}


def batchable(fn: Callable) -> Callable:
    """Decorator to allow modules to accept batch or single board input."""

    @wraps(fn)
    def wrapper(boards: np.ndarray, *args, **kwargs) -> np.ndarray:
        boards = np.asarray(boards)
        if boards.ndim == 2:
            return fn(boards, *args, **kwargs)
        return np.stack([fn(b, *args, **kwargs) for b in boards], axis=0)

    return wrapper


def safe_call(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call func filtering out kwargs not in its signature."""
    sig = inspect.signature(inspect.unwrap(func))
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **allowed)


# Logging configuration
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure default logging if no handlers are present."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )


# 來自 log_once_patch.txt 和 final_numpy_coord_fix_summary.txt
_seen_modules_once = set()


# 來自 final_numpy_coord_fix_summary.txt
def _to_native_coord(coord):
    r, c = coord
    return int(r), int(c)


# Math helpers
class MathUtils:
    """Utility functions for common mathematical operations."""

    def sigmoid(self, x: float, k: float = 1.0) -> float:
        """Clamped sigmoid to avoid overflow."""
        try:
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x))
        except OverflowError:
            return 0.0 if -k * x > 0 else 1.0

    def normalize_value(
        self, value: float, min_val: float, max_val: float, clamp: bool = True
    ) -> float:
        """Normalize value to [0, 1]."""
        if math.isclose(max_val, min_val, rel_tol=1e-9):
            return (
                0.5
                if math.isclose(value, min_val, rel_tol=1e-9)
                else (0.0 if value < min_val else 1.0)
            )
        normalized = (value - min_val) / (max_val - min_val + 1e-10)
        return max(0.0, min(1.0, normalized)) if clamp else normalized

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two (row, col) points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# Board analysis helpers
class BoardAnalyzerUtils:
    """Utility collection for scratch-card grid analysis."""

    def get_neighborhood_values(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        radius: int = 2,
        connectivity: int = 8,
        val_func: Callable[[int], Optional[float]] = lambda x: (
            float(x) if x != -1 else None
        ),
        include_center: bool = False,
        **kw,
    ) -> List[float]:
        """
        Collect values surrounding grid[r, c] in a square radius.

        Accepts legacy keywords:
            eight_connectivity / four_connectivity
        """
        if "eight_connectivity" in kw:
            connectivity = 8 if kw.pop("eight_connectivity") else 4
        if "four_connectivity" in kw:
            connectivity = 4 if kw.pop("four_connectivity") else 8

        neighbors: List[float] = []
        rows, cols = grid.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_center and dr == 0 and dc == 0:
                    continue
                if connectivity == 4 and abs(dr) + abs(dc) > radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    processed_val = val_func(grid[nr, nc])
                    if processed_val is not None:
                        neighbors.append(processed_val)
        return neighbors

    def check_sequences(
        self,
        board: np.ndarray,
        original_grid: np.ndarray,
        min_len: int = 3,
        allow_gaps: int = 1,
    ) -> bool:
        """Return True if board contains arithmetic/geometric sequence in various shapes."""
        rows, cols = board.shape
        shapes = [
            lambda r, c: [(r + i, c) for i in range(min_len)],  # 行
            lambda r, c: [(r, c + i) for i in range(min_len)],  # 列
            lambda r, c: [(r + i, c + i) for i in range(min_len)],  # 主對角線
            lambda r, c: [(r + i, c - i) for i in range(min_len)],  # 副對角線
            lambda r, c: [(r, c), (r + 1, c + 2), (r + 2, c + 1)],  # Z 型
            lambda r, c: [(r + i, c) for i in range(2)] + [(r + 2, c + 2)],  # L 型
        ]

        for r in range(rows):
            for c in range(cols):
                if board[r, c] == -1:
                    continue
                for shape_gen in shapes:
                    points = [
                        (rr, cc)
                        for rr, cc in shape_gen(r, c)
                        if 0 <= rr < rows and 0 <= cc < cols
                    ]
                    if len(points) >= min_len:
                        values = [
                            board[rr, cc] for rr, cc in points if board[rr, cc] != -1
                        ]
                        if len(
                            values
                        ) >= min_len and self.get_arithmetic_or_geometric_sequences(
                            np.array(values), min_len, allow_gaps
                        ):
                            return True
        return False

    def get_arithmetic_or_geometric_sequences(
        self,
        line: np.ndarray,
        min_len: int = 3,
        allow_gaps: int = 1,
    ) -> List[List[int]]:
        """Detect arithmetic/geometric subsequences in a 1-D array."""
        sequences: List[List[int]] = []
        n = len(line)
        for i in range(n):
            if line[i] == -1:
                continue
            for j in range(i + 1, n):
                if line[j] == -1:
                    temp_gap = 0
                    for k in range(j, n):
                        if line[k] == -1:
                            temp_gap += 1
                        else:
                            if temp_gap <= allow_gaps:
                                diff = line[k] - line[i]
                                if diff == 0:
                                    break
                                seq_vals = [line[i], line[k]]
                                gap_cnt = temp_gap
                                for idx_l in range(k + 1, n):
                                    if line[idx_l] == -1:
                                        gap_cnt += 1
                                        if gap_cnt > allow_gaps:
                                            break
                                        continue
                                    expected = seq_vals[-1] + diff
                                    if math.isclose(
                                        line[idx_l], expected, rel_tol=1e-9
                                    ):
                                        seq_vals.append(line[idx_l])
                                        gap_cnt = 0
                                    else:
                                        break
                                if len(seq_vals) >= min_len:
                                    sequences.append(seq_vals)
                            break
                else:
                    diff = line[j] - line[i]
                    if diff == 0:
                        continue
                    seq_vals = [line[i], line[j]]
                    gap_cnt = 0
                    for k in range(j + 1, n):
                        if line[k] == -1:
                            gap_cnt += 1
                            if gap_cnt > allow_gaps:
                                break
                            continue
                        expected = seq_vals[-1] + diff
                        if math.isclose(line[k], expected, rel_tol=1e-9):
                            seq_vals.append(line[k])
                            gap_cnt = 0
                        else:
                            break
                    if len(seq_vals) >= min_len:
                        sequences.append(seq_vals)
        return sequences

    def get_card_max_value_from_gridDimensions(
        self, grid_shape: Tuple[int, int]
    ) -> int:
        """Return rows×cols (max possible face value)."""
        rows, cols = grid_shape
        return rows * cols if rows and cols else 0

    def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
        """Return unused numbers > 0 that can still appear on the board."""
        rows, cols = grid.shape
        all_vals = set(
            range(1, self.get_card_max_value_from_gridDimensions((rows, cols)) + 1)
        )
        used = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
        return all_vals - used


# 來自 fix_unsupported_itemsize_and_cache.txt
DTYPE_DEFAULT = np.int32
ITEMSIZE = np.dtype(DTYPE_DEFAULT).itemsize  # 修正為 4 bytes


def bytes_to_grid(grid_bytes: bytes, shape):
    # zero-copy, read-only view
    arr = np.frombuffer(grid_bytes, dtype=DTYPE_DEFAULT)
    return arr.reshape(shape)


# Module registry
REGISTERED_MODULES_BRAIN: Dict[
    str, Callable[[np.ndarray, Optional[str]], np.ndarray]
] = {}

if os.getenv("ENABLE_LEGACY", "0") != "1":
    logger.info(
        "Legacy modules disabled by default (ENABLE_LEGACY=0). Running Q1-Q4 only."
    )
else:
    logger.warning(
        "Legacy modules enabled via ENABLE_LEGACY=1. Performance may degrade by 30-40%."
    )


# ----------------------------------------------------------------------
# Utility kernels / helpers from q_series_advanced_patch.py
# ----------------------------------------------------------------------
def _local_hist(grid, bins=100, win=5):
    """Return histogram for each cell by scanning the entire board."""

    rows, cols = grid.shape
    hist = np.zeros((rows, cols, bins), dtype=float)
    flat = np.mod(grid, bins).ravel()
    global_hist = np.bincount(flat, minlength=bins).astype(float)

    for r in range(rows):
        for c in range(cols):
            hist[r, c] = global_hist

    hist = hist / hist.sum(-1, keepdims=True).clip(1e-9)
    return hist


# ----------------------------------------------------------------------
# Q-Series modules from q_series_advanced_patch.py
# ----------------------------------------------------------------------
def compute_global_features(grid: np.ndarray, bins: int = 100):
    """
    Robust global stats: mean, std, entropy (0‑1), and PDF (bins).
    Works even when grid contains negative or very large integers.
    """
    flat = grid.ravel().astype(float)
    mean_ = flat.mean()
    std_ = flat.std(ddof=0)

    # Scale values into 0..bins‑1
    minv, maxv = flat.min(), flat.max()
    if minv == maxv:
        # Degenerate board – zero entropy
        p = np.zeros(bins, dtype=float)
        p[0] = 1.0
        return mean_, std_, 0.0, p

    norm = (flat - minv) / (maxv - minv)
    idx = np.clip((norm * (bins - 1)).astype(int), 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(float) + 1e-9
    p = counts / counts.sum()

    entropy = -(p * np.log2(p)).sum() / np.log2(bins)  # 0‑1
    return mean_, std_, entropy, p


@batchable
def EXT_Q5_GlobalEntropy_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    _, _, entropy, _ = compute_global_features(grid)
    vals = grid.ravel().astype(float)
    if len(np.unique(vals)) < 2:
        centroids = np.array([vals[0], vals[0]])
        labels = np.zeros_like(vals, dtype=int)
    else:
        centroids, labels = kmeans2(vals, k=2, minit="points")
    hot = int(np.argmax(centroids))
    coords = np.column_stack(np.unravel_index(np.arange(vals.size), grid.shape))
    hot_coords = coords[labels == hot]
    heat_score = (
        0.0
        if hot_coords.size == 0
        else 1.0
        - (
            np.linalg.norm(hot_coords - hot_coords.mean(0), axis=1).mean()
            / max(grid.shape)
        )
    )
    return np.full(grid.shape, 0.6 * entropy + 0.4 * heat_score, dtype=float)


def compute_line_bridge_score(grid):
    """Return bridge confidence by enumerating all adjacent pairs."""

    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and grid[r, c] == grid[r, c + 1]:
                score[r, c] += 1.0
                score[r, c + 1] += 1.0
            if r + 1 < rows and grid[r, c] == grid[r + 1, c]:
                score[r, c] += 1.0
                score[r + 1, c] += 1.0

            val = grid[r, c]
            matches = 0.0
            if r > 0 and grid[r - 1, c] == val:
                matches += 1.0
            if r + 1 < rows and grid[r + 1, c] == val:
                matches += 1.0
            if c > 0 and grid[r, c - 1] == val:
                matches += 1.0
            if c + 1 < cols and grid[r, c + 1] == val:
                matches += 1.0
            score[r, c] += matches / 4.0

    mx = score.max(initial=1.0)
    score /= mx
    return score


@batchable
def EXT_Q6_LineBridge_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    return compute_line_bridge_score(grid)


def compute_local_variance_prior(grid, w=3):
    """Variance prior computed against all cells instead of local windows."""

    rows, cols = grid.shape
    g = grid.astype(float)
    score = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            diffs = (g[r, c] - g) ** 2
            score[r, c] = 1.0 / (diffs.mean() + 1e-6)

    mn, mx = score.min(), score.max()
    if mx > mn:
        score = (score - mn) / (mx - mn)
    else:
        score.fill(0.0)
    return score


@batchable
def EXT_Q7_VariancePrior_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    _, std_, _, _ = compute_global_features(grid)
    alpha = min(std_ / 15, 1)
    return alpha * compute_local_variance_prior(grid) + (1 - alpha) * 0.5


@batchable
def EXT_Q8_SpatialKL_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A", win=5
) -> np.ndarray:
    _, _, _, global_p = compute_global_features(grid)
    local_hist = _local_hist(grid, bins=100, win=win)  # (r,c,b)
    kl = (local_hist * np.log((local_hist + 1e-9) / global_p)).sum(-1)
    kl = (kl - kl.min()) / (kl.max() - kl.min() + 1e-9)
    return 1 - kl  # high = match global, anomalies low


@batchable
def EXT_Q9_MultiScaleEntropy_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    ent1 = EXT_Q5_GlobalEntropy_Vec(grid)  # global
    ent2 = (
        EXT_Q5_GlobalEntropy_Vec(grid[::2, ::2], request_id)
        .repeat(2, 0)
        .repeat(2, 1)[: grid.shape[0], : grid.shape[1]]
    )
    ent3 = (
        EXT_Q5_GlobalEntropy_Vec(grid[::4, ::4], request_id)
        .repeat(4, 0)
        .repeat(4, 1)[: grid.shape[0], : grid.shape[1]]
    )
    return 0.5 * ent1 + 0.3 * ent2 + 0.2 * ent3


@batchable
def EXT_Q10_DistPotential_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    target_mask = grid == grid.max()  # treat max value as revealed?
    dist = ndi.distance_transform_edt(~target_mask)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
    return 1 - dist_norm


@batchable
def EXT_Q11_GlobalDigitAffinity_Vec(
    grid: np.ndarray, target: Optional[int] = None, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    if target is None:
        return np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    max_val = rows * cols
    vals = np.arange(1, max_val + 1)

    sim = np.zeros_like(vals, dtype=float)
    np.maximum(sim, (vals % 10 == target % 10).astype(float) * 1.0, out=sim)
    np.maximum(
        sim, np.isin(np.abs(vals - target), [10, 20]).astype(float) * 0.7, out=sim
    )
    np.maximum(sim, (vals % 10 == target // 10).astype(float) * 0.4, out=sim)

    rr = np.arange(rows)[:, None]
    cc = np.arange(cols)[None, :]
    center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
    dist = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    kernel = 1.0 / (1.0 + 0.5 * dist)

    masks = (grid[..., None] == vals).astype(float)
    k_fft = rfftn(kernel, s=grid.shape)
    masks_fft = rfftn(masks, s=grid.shape, axes=(0, 1))
    conv = irfftn(masks_fft * k_fft[..., None], s=grid.shape, axes=(0, 1))
    score = np.tensordot(conv, sim, axes=([2], [0]))

    score = np.where(grid == -1, score, 0.0)
    mn, mx = score.min(), score.max()
    if mx > mn:
        score = (score - mn) / (mx - mn)
    else:
        score.fill(0.0)
    return score.astype(np.float32)


@lru_cache(maxsize=32)
def _get_window(rows: int, cols: int, win_fn: str) -> np.ndarray:
    if win_fn == "hann":
        wr = np.hanning(rows)[:, None]
        wc = np.hanning(cols)[None, :]
        return wr * wc
    if win_fn == "hamming":
        wr = np.hamming(rows)[:, None]
        wc = np.hamming(cols)[None, :]
        return wr * wc
    return np.ones((rows, cols), dtype=float)


@batchable
def EXT_Q13_GlobalConsistencySpectrum_Vec(
    grid: np.ndarray,
    *,
    WIN_FN: str = "hann",
    TRIM_LOW_FREQ: int = 1,
    ENERGY_THRESH: float = 0.2,
    GAP_FUNC: Optional[Callable[[int], float]] = None,
    request_id: Optional[str] = "N/A",
) -> np.ndarray:
    """Score cells by spectrum consistency of known-number positions."""

    _ = GAP_FUNC  # compatibility
    mask = (grid != -1).astype(float)
    rows, cols = mask.shape

    win = _get_window(rows, cols, WIN_FN)
    masked = mask * win
    masked = masked - masked.mean()
    spec = rfftn(masked, s=masked.shape, axes=(0, 1))

    trim = min(TRIM_LOW_FREQ, rows // 4, cols // 4)
    if trim > 0:
        spec[: trim + 1, :] = 0
        spec[:, : trim + 1] = 0

    energy = np.abs(spec) ** 2
    max_e = energy.max(initial=0.0)
    if max_e <= 0:
        score = np.zeros_like(mask)
    else:
        mask_high = energy >= ENERGY_THRESH * max_e
        recon = irfftn(spec * mask_high, s=masked.shape, axes=(0, 1))
        score = np.abs(recon)
        mn, mx = score.min(), score.max()
        if mx > mn:
            score = (score - mn) / (mx - mn)
        else:
            score.fill(0.0)

    score = np.where(grid == -1, score, 0.0)
    return score.astype(np.float32)


def _shift(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(arr)
    r_start = max(0, dr)
    r_end = arr.shape[0] + min(0, dr)
    c_start = max(0, dc)
    c_end = arr.shape[1] + min(0, dc)
    out[r_start:r_end, c_start:c_end] = arr[
        r_start - dr : r_end - dr, c_start - dc : c_end - dc
    ]
    return out


@batchable
def EXT_Q12_ArithmeticProgression_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    rows, cols = grid.shape
    val = np.where(grid == -1, 0, grid)
    mask = (grid != -1).astype(int)
    score = np.zeros_like(val, dtype=float)

    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (2, 1),
        (2, -1),
        (1, 2),
        (-1, 2),
    ]

    for dr, dc in directions:
        prev = _shift(val, -dr, -dc)
        next_ = _shift(val, dr, dc)
        conv1 = prev - 2 * val + next_
        cnt = _shift(mask, -dr, -dc) + mask + _shift(mask, dr, dc)
        valid = (conv1 == 0) & (cnt == 3)
        w_gap = 1.0 / (1.0 + max(abs(dr), abs(dc)))
        v = valid.astype(float)
        score += w_gap * (v + _shift(v, dr, dc) + _shift(v, -dr, -dc))

    score = np.where(grid == -1, score, 0.0)
    mn, mx = score.min(), score.max()
    if mx > mn:
        score = (score - mn) / (mx - mn)
    else:
        score.fill(0.0)
    return score.astype(np.float32)


# --- Scoring Module Implementations ---


@batchable
def EXT_M1_Tail_Pattern_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score based on tail number patterns in 5x5 neighborhood."""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    utils = BoardAnalyzerUtils()
    radius = min(2, min(rows, cols) // 2 - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            neighbors = utils.get_neighborhood_values(
                grid, r, c, radius=radius, eight_connectivity=True
            )
            if not neighbors:
                continue
            tail_counts = Counter(int(v % 10) for v in neighbors if v > 0)
            total_tails = sum(tail_counts.values()) or 1e-10
            legal_values = utils.get_legal_values_for_placement(grid)
            max_score = 0.0
            mean_val = np.mean([v for v in grid[grid != -1] if v > 0]) or 1.0
            for val in legal_values:
                tail = val % 10
                base_score = tail_counts.get(tail, 0) / total_tails
                distance_factor = 1.0 - (abs(val - mean_val) % 10) * 0.05
                score = base_score * distance_factor + random.uniform(0, 0.1)
                max_score = max(max_score, MathUtils().normalize_value(score, 0, 1.0))
            scores[r, c] = max_score
    return scores


@batchable
def EXT_M3_Local_Focus_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score based on 5x5 neighborhood mean and variance."""
    rows, cols = grid.shape
    radius = min(2, min(rows, cols) // 2 - 1)
    if radius <= 0:
        return np.zeros_like(grid, dtype=float)

    size = 2 * radius + 1
    g = grid.astype(float)
    mean = ndi.uniform_filter(g, size=size, mode="reflect")
    var = ndi.uniform_filter(g**2, size=size, mode="reflect") - mean**2
    std = np.sqrt(var, dtype=float)
    deviation = np.abs(g - mean) / (std + 1e-6)
    norm = (deviation - deviation.min()) / (deviation.max() - deviation.min() + 1e-9)
    scores = np.where(grid == -1, norm, 0.0)
    return scores


@batchable
def EXT_M10_Sequence_Block_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score based on sequence blocks in 5x5 neighborhood."""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    utils = BoardAnalyzerUtils()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            row_block = grid[max(0, r - 2) : min(rows, r + 3)]
            col_block = grid[:, max(0, c - 2) : min(cols, c + 3)]
            row_seqs = []
            for i in range(row_block.shape[0]):
                row_seqs.extend(
                    utils.get_arithmetic_or_geometric_sequences(row_block[i])
                )
            col_seqs = []
            for i in range(col_block.shape[1]):
                col_seqs.extend(
                    utils.get_arithmetic_or_geometric_sequences(col_block[:, i])
                )
            diag_seqs = []
            sub_grid = grid[
                max(0, r - 2) : min(rows, r + 3), max(0, c - 2) : min(cols, c + 3)
            ]
            for offset in range(
                -min(sub_grid.shape[0], sub_grid.shape[1]),
                min(sub_grid.shape[0], sub_grid.shape[1]),
            ):
                diag = np.diagonal(sub_grid, offset)
                if len(diag) >= min(3, len(diag)):
                    diag_seqs.extend(utils.get_arithmetic_or_geometric_sequences(diag))
                diag_flipped = np.diagonal(np.fliplr(sub_grid), offset)
                if len(diag_flipped) >= min(3, len(diag_flipped)):
                    diag_seqs.extend(
                        utils.get_arithmetic_or_geometric_sequences(diag_flipped)
                    )
            legal_values = utils.get_legal_values_for_placement(grid)
            max_score = 0.0
            for val in legal_values:
                row_fit = any(val in seq for seq in row_seqs)
                col_fit = any(val in seq for seq in col_seqs)
                diag_fit = any(val in seq for seq in diag_seqs)
                score = (row_fit + col_fit + diag_fit) / 3.0
                max_score = max(max_score, MathUtils().normalize_value(score, 0, 1.0))
            scores[r, c] = max_score
    return scores


error_memory = defaultdict(Counter)


@batchable
def EXT_R3_Error_Correction_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score based on historical error correction in 5x5 neighborhood."""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    utils = BoardAnalyzerUtils()
    legal_values = utils.get_legal_values_for_placement(grid)
    radius = min(2, min(rows, cols) // 2 - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            base_score = 0.5
            for val in legal_values:
                error_count = error_memory[(r, c)][val]
                for nr, nc in [
                    (r + dr, c + dc)
                    for dr in range(-radius, radius + 1)
                    for dc in range(-radius, radius + 1)
                    if 0 <= r + dr < rows and 0 <= c + dc < cols
                ]:
                    error_count += error_memory[(nr, nc)][val] * 0.1
                penalty = min(0.3, error_count * 0.05)
                score = MathUtils().normalize_value(base_score - penalty, 0, 1.0)
                if score > scores[r, c]:
                    scores[r, c] = score
    return scores


@batchable
def EXT_F7_Strong_Pattern_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score based on strong arithmetic or symmetry patterns."""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    utils = BoardAnalyzerUtils()

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            row_seq = utils.check_sequences(
                grid[r : r + 1], grid, min_len=3, allow_gaps=1
            )
            col_seq = utils.check_sequences(
                grid[:, c : c + 1].T, grid, min_len=3, allow_gaps=1
            )
            symmetry = r == cols - 1 - c or c == rows - 1 - r
            legal_values = utils.get_legal_values_for_placement(grid)
            max_score = 0.0
            for val in legal_values:
                base_score = 0.5
                if row_seq or col_seq:
                    base_score += 0.3
                if (
                    symmetry
                    and (0 <= rows - 1 - r < rows and 0 <= cols - 1 - c < cols)
                    and grid[rows - 1 - r, cols - 1 - c] == val
                ):
                    base_score += 0.2
                score = MathUtils().normalize_value(base_score, 0, 1.0)
                max_score = max(max_score, score)
            scores[r, c] = max_score
    return scores


@batchable
def EXT_M11_Mirror_Sequence_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Penalty for sequential pairs across mirror-symmetric points."""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            r2, c2 = rows - 1 - r, cols - 1 - c
            if r > r2 or (r == r2 and c >= c2):
                continue
            if abs(int(grid[r, c]) - int(grid[r2, c2])) == 1:
                scores[r, c] += 1.0
                scores[r2, c2] += 1.0

    if scores.max(initial=0.0) > 0:
        scores /= scores.max()
    return scores.astype(np.float32)


@batchable
def EXT_GM20_Skip_Pattern_Confidence_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score based on skip pattern confidence."""
    rows, cols = grid.shape
    scores = np.zeros((rows, cols), dtype=float)
    revealed = [
        {"value": int(grid[r, c]), "r": r, "c": c}
        for r in range(rows)
        for c in range(cols)
        if grid[r, c] != -1 and grid[r, c] > 0
    ]
    if not revealed:
        return scores

    utils = BoardAnalyzerUtils()
    math_utils = MathUtils()
    max_val = utils.get_card_max_value_from_gridDimensions((rows, cols))
    base_pos = {k: ((k - 1) // cols, (k - 1) % cols) for k in range(1, max_val + 1)}
    skip_vecs = {
        info["value"]: (
            info["r"] - base_pos[info["value"]][0],
            info["c"] - base_pos[info["value"]][1],
        )
        for info in revealed
        if info["value"] in base_pos
    }
    if not skip_vecs:
        return scores

    counts = Counter(skip_vecs.values())
    min_occ = max(1, int(len(skip_vecs) * 0.05))
    dominant_patterns: List[Dict[str, Any]] = []
    for vec, cnt in counts.most_common():
        if cnt < min_occ:
            break
        pattern_vals = sorted(v for v, sv in skip_vecs.items() if sv == vec)
        strength = math_utils.normalize_value(cnt, min_occ, len(skip_vecs)) * 1.1
        dominant_patterns.append(
            {"skip": vec, "values": pattern_vals, "strength": strength}
        )

    if not dominant_patterns:
        return scores

    legal_nums = utils.get_legal_values_for_placement(grid)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            best_conf = 0.0
            for num in legal_nums:
                if num not in base_pos:
                    continue
                base_r, base_c = base_pos[num]
                for pat in dominant_patterns:
                    dr, dc = pat["skip"]
                    if base_r + dr == r and base_c + dc == c:
                        enh = 0.5
                        if len(pat["values"]) >= 1:
                            seq = sorted(pat["values"] + [num])
                            if len(seq) >= 2 and len(set(np.diff(seq))) == 1:
                                enh += 0.5
                            elif len(seq) >= 3 and min(seq) < num < max(seq):
                                enh += 0.15
                        best_conf = max(best_conf, pat["strength"] * enh)
            scores[r, c] = math_utils.normalize_value(best_conf, 0, 1.0)
    return scores


@batchable
def EXT_Q1_ProximityEntropy_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score cells by comparing their distance profile to global distribution."""

    rows, cols = grid.shape
    known = np.argwhere(grid != -1)
    if known.size == 0:
        return np.zeros((rows, cols), dtype=float)

    # Global distance distribution among all revealed cells
    if known.shape[0] > 1:
        diff = known[:, None, :] - known[None, :, :]
        dist_pairs = np.abs(diff).sum(-1)
        dist_pairs = dist_pairs[np.triu_indices(dist_pairs.shape[0], k=1)]
        max_d = rows + cols
        hist_global = np.bincount(dist_pairs.ravel(), minlength=max_d + 1).astype(float)
    else:
        max_d = rows + cols
        hist_global = np.zeros(max_d + 1, dtype=float)
        hist_global[1] = 1.0
    hist_global /= hist_global.sum() + 1e-9

    def _cell_hist(r: int, c: int) -> np.ndarray:
        dists = np.abs(known[:, 0] - r) + np.abs(known[:, 1] - c)
        h = np.bincount(dists, minlength=max_d + 1).astype(float)
        return h / (h.sum() + 1e-9)

    score = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != -1:
                continue
            hist_cell = _cell_hist(r, c)
            kl = np.sum(hist_cell * np.log((hist_cell + 1e-9) / (hist_global + 1e-9)))
            score[r, c] = 1.0 - kl

    mn, mx = score.min(), score.max()
    if mx > mn:
        score = (score - mn) / (mx - mn)
    else:
        score.fill(0.0)
    return score


@batchable
def EXT_Q2_PotentialPath_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score cells by enumerating sequential neighbors across the board."""

    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)
    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (0, -1),
        (-1, 0),
        (-1, -1),
    ]

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -1:
                continue
            val = int(grid[r, c])
            for dr, dc in directions:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols and grid[r2, c2] != -1:
                    if abs(val - int(grid[r2, c2])) == 1:
                        score[r, c] += 1.0
                        score[r2, c2] += 1.0

    mx = score.max(initial=0.0)
    if mx > 0:
        score /= mx
    return score


@batchable
def EXT_Q3_DiscontinuitySym_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score symmetry by scanning all mirror pairs on the board."""

    rows, cols = grid.shape
    score = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            r2, c2 = rows - 1 - r, cols - 1 - c
            if r > r2 or (r == r2 and c >= c2):
                continue
            if grid[r, c] != -1 and grid[r2, c2] != -1 and grid[r, c] == grid[r2, c2]:
                score[r, c] += 1.0
                score[r2, c2] += 1.0

    mx = score.max(initial=0.0)
    if mx > 0:
        score /= mx
    return score


@batchable
def EXT_Q4_ControlComposite_Vec(
    grid: np.ndarray, target: Optional[int] = None, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Weighted combination of Q1-Q7 style modules over the whole board."""

    modules = [
        "EXT_Q1_ProximityEntropy_Vec",
        "EXT_Q2_PotentialPath_Vec",
        "EXT_Q3_DiscontinuitySym_Vec",
        "EXT_Q5_GlobalEntropy_Vec",
        "EXT_Q6_LineBridge_Vec",
        "EXT_Q7_VariancePrior_Vec",
    ]
    stack = np.stack(
        [
            get_module_score(m, grid, target=target, request_id=request_id)
            for m in modules
        ],
        axis=0,
    )
    weights = np.array([AGG_WEIGHTS.get(m, 1.0) for m in modules], dtype=float)
    return aggregate_scores(stack, weights, modules)


@batchable
def EXT_M12_RestoreOriginalValue_Vec(
    grid: np.ndarray,
    *,
    original_grid: Optional[np.ndarray] = None,
    request_id: Optional[str] = "N/A",
) -> np.ndarray:
    """Boost cells that were removed from the original grid."""
    if original_grid is None:
        return np.zeros_like(grid, dtype=float)
    mask = (original_grid != -1) & (grid == -1)
    score = mask.astype(float)
    mx = score.max(initial=0.0)
    if mx > 0:
        score /= mx
    return score.astype(np.float32)


@batchable
def EXT_Q14_TargetAffinity_Vec(
    grid: np.ndarray,
    *,
    target: Optional[int] = None,
    priors: Optional[Dict[int, float]] = None,
    request_id: Optional[str] = "N/A",
) -> np.ndarray:
    """Return uniform score based on prior affinity for ``target``."""
    source = priors if priors is not None else globals().get("priors", {})
    affinity = source.get(int(target), 0.0) if target is not None else 0.0
    rows, cols = grid.shape
    return np.full((rows, cols), float(affinity), dtype=float)


@batchable
def EXT_Q15_GlobalSpread_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Prefer central blanks to encourage global coverage."""
    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)
    if blanks.size == 0:
        return np.zeros((rows, cols), dtype=float)
    center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
    score = np.zeros((rows, cols), dtype=float)
    for r, c in blanks:
        dist = math.hypot(r - center[0], c - center[1])
        score[r, c] = 1.0 / (1.0 + dist)
    mx = score.max(initial=0.0)
    if mx > 0:
        score /= mx
    return score.astype(np.float32)


@batchable
def EXT_Q16_NumericalRelationalPattern_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Score cells via combined numerical relational patterns."""
    rows, cols = grid.shape
    blanks = np.argwhere(grid == -1)
    if blanks.size == 0:
        return np.zeros((rows, cols), dtype=float)

    score = np.zeros((rows, cols), dtype=float)
    for r, c in blanks:
        mirror = 0.0
        checks = 0
        if 0 <= rows - 1 - r < rows:
            if grid[rows - 1 - r, c] != -1:
                mirror += 1.0
            checks += 1
        if 0 <= cols - 1 - c < cols:
            if grid[r, cols - 1 - c] != -1:
                mirror += 1.0
            checks += 1
        if 0 <= rows - 1 - r < rows and 0 <= cols - 1 - c < cols:
            if grid[rows - 1 - r, cols - 1 - c] != -1:
                mirror += 1.0
            checks += 1
        if rows == cols and 0 <= c < rows and 0 <= r < cols:
            if grid[c, r] != -1:
                mirror += 1.0
            checks += 1
        mirror_score = mirror / float(checks or 1)

        seq_score = 0.0
        seq_checks = 0
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            nnr, nnc = nr + dr, nc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                seq_checks += 1
                if 0 <= nnr < rows and 0 <= nnc < cols and grid[nnr, nnc] != -1:
                    if abs(int(grid[nnr, nnc]) - int(grid[nr, nc])) == 1:
                        seq_score += 1.0
        seq_score /= float(seq_checks or 1)

        jump_score = 0.0
        jump_checks = 0
        for dr, dc in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                jump_score += 1.0
            jump_checks += 1
        jump_score /= float(jump_checks or 1)

        neighbors = [
            int(grid[nr, nc])
            for dr in range(-1, 2)
            for dc in range(-1, 2)
            if not (dr == 0 and dc == 0)
            for nr, nc in [(r + dr, c + dc)]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1
        ]
        tail_score = 0.0
        unique_score = 0.0
        if neighbors:
            tails = [v % 10 for v in neighbors]
            tail_score = len(set(tails)) / float(len(tails))
            unique_score = len(set(neighbors)) / float(len(neighbors))

        features = [mirror_score, seq_score, jump_score, tail_score, unique_score]
        score[r, c] = sum(features) / len(features)

    mx = score.max(initial=0.0)
    if mx > 0:
        score /= mx
    return score.astype(np.float32)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------
mods = {
    "EXT_Q5_GlobalEntropy_Vec": EXT_Q5_GlobalEntropy_Vec,
    "EXT_Q6_LineBridge_Vec": EXT_Q6_LineBridge_Vec,
    "EXT_Q7_VariancePrior_Vec": EXT_Q7_VariancePrior_Vec,
    "EXT_Q8_SpatialKL_Vec": EXT_Q8_SpatialKL_Vec,
    "EXT_Q9_MultiScaleEntropy_Vec": EXT_Q9_MultiScaleEntropy_Vec,
    "EXT_Q10_DistPotential_Vec": EXT_Q10_DistPotential_Vec,
    "EXT_Q11_GlobalDigitAffinity_Vec": EXT_Q11_GlobalDigitAffinity_Vec,
    "EXT_Q12_ArithmeticProgression_Vec": EXT_Q12_ArithmeticProgression_Vec,
    "EXT_Q13_GlobalConsistencySpectrum_Vec": EXT_Q13_GlobalConsistencySpectrum_Vec,
    "EXT_M12_RestoreOriginalValue_Vec": EXT_M12_RestoreOriginalValue_Vec,
    "EXT_Q14_TargetAffinity_Vec": EXT_Q14_TargetAffinity_Vec,
    "EXT_Q15_GlobalSpread_Vec": EXT_Q15_GlobalSpread_Vec,
    "EXT_Q16_NumericalRelationalPattern_Vec": EXT_Q16_NumericalRelationalPattern_Vec,
    "EXT_M1_Tail_Pattern_Vec": EXT_M1_Tail_Pattern_Vec,
    "EXT_M3_Local_Focus_Vec": EXT_M3_Local_Focus_Vec,
    "EXT_M10_Sequence_Block_Vec": EXT_M10_Sequence_Block_Vec,
    "EXT_R3_Error_Correction_Vec": EXT_R3_Error_Correction_Vec,
    "EXT_F7_Strong_Pattern_Vec": EXT_F7_Strong_Pattern_Vec,
    "EXT_M11_Mirror_Sequence_Vec": EXT_M11_Mirror_Sequence_Vec,
    "EXT_GM20_Skip_Pattern_Confidence_Vec": EXT_GM20_Skip_Pattern_Confidence_Vec,
    "EXT_Q1_ProximityEntropy_Vec": EXT_Q1_ProximityEntropy_Vec,
    "EXT_Q2_PotentialPath_Vec": EXT_Q2_PotentialPath_Vec,
    "EXT_Q3_DiscontinuitySym_Vec": EXT_Q3_DiscontinuitySym_Vec,
    "EXT_Q4_ControlComposite_Vec": EXT_Q4_ControlComposite_Vec,
    "GlobalOffsetCooccurrence": lambda boards, target, offsets=None, **_: global_offset_cooccurrence(
        boards, target, offsets
    ),
    "ValueProximityDistribution": lambda boards, target, tolerance=1, radius=1, **_: neighbor_value_distribution(
        boards, target, tolerance, radius
    ),
}
try:
    REGISTERED_MODULES_BRAIN
except NameError:
    REGISTERED_MODULES_BRAIN = {}
REGISTERED_MODULES_BRAIN.update(mods)

FAST_PHASE = [
    "EXT_Q1_ProximityEntropy_Vec",
    "EXT_Q2_PotentialPath_Vec",
    "EXT_Q5_GlobalEntropy_Vec",
    "EXT_Q8_SpatialKL_Vec",
    "EXT_M12_RestoreOriginalValue_Vec",
    "EXT_Q15_GlobalSpread_Vec",
]

RERANK_PHASE = [
    "EXT_Q3_DiscontinuitySym_Vec",
    "EXT_Q6_LineBridge_Vec",
    "EXT_Q7_VariancePrior_Vec",
    "EXT_Q9_MultiScaleEntropy_Vec",
    "EXT_Q10_DistPotential_Vec",
    "EXT_Q14_TargetAffinity_Vec",
    "EXT_M11_Mirror_Sequence_Vec",
]

# Static weights for module aggregation
AGG_WEIGHTS = {
    "EXT_Q1_ProximityEntropy_Vec": 0.20,
    "EXT_Q2_PotentialPath_Vec": 0.15,
    "EXT_Q3_DiscontinuitySym_Vec": 0.10,
    "EXT_Q4_ControlComposite_Vec": 0.10,
    "EXT_Q5_GlobalEntropy_Vec": 0.08,
    "EXT_Q6_LineBridge_Vec": 0.08,
    "EXT_Q7_VariancePrior_Vec": 0.08,
    "EXT_Q8_SpatialKL_Vec": 0.05,
    "EXT_Q9_MultiScaleEntropy_Vec": 0.05,
    "EXT_Q10_DistPotential_Vec": 0.04,
    "EXT_Q11_GlobalDigitAffinity_Vec": 0.03,
    "EXT_Q12_ArithmeticProgression_Vec": 0.04,
    "EXT_Q13_GlobalConsistencySpectrum_Vec": 0.04,
    "EXT_M12_RestoreOriginalValue_Vec": 0.05,
    "EXT_Q14_TargetAffinity_Vec": 0.05,
    "EXT_Q15_GlobalSpread_Vec": 0.04,
    "EXT_Q16_NumericalRelationalPattern_Vec": 0.05,
    "EXT_M11_Mirror_Sequence_Vec": -0.03,
}


def _load_weights() -> Dict[str, float]:
    """Return normalized weights with ENV overrides."""
    w = AGG_WEIGHTS.copy()
    for name in w:
        env_key = f"WEIGHT_{name.split('_')[1]}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                w[name] = float(env_val)
            except ValueError:
                logger.warning("Invalid weight for %s: %s", env_key, env_val)
    total = sum(w.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        logger.info("Normalizing module weights (sum %.3f)", total)
        for k in w:
            w[k] /= total or 1e-10
    return w


AGG_WEIGHTS = _load_weights()


def get_core_modules(limit: int | None = None) -> list[str]:
    """Return top-N modules ranked by weight.

    The limit defaults to the ``CORE_LIMIT`` environment variable (8) and is
    clamped to the range 5–10.
    """
    try:
        limit_env = int(os.getenv("CORE_LIMIT", "8"))
    except ValueError:  # FIXME invalid env value
        logger.warning(
            "Invalid CORE_LIMIT '%s', using default 8",
            os.getenv("CORE_LIMIT"),
        )
        limit_env = 8
    if limit is None:
        limit = limit_env
    limit = max(5, min(10, limit))
    sorted_mods = sorted(AGG_WEIGHTS.items(), key=lambda kv: kv[1], reverse=True)
    return [m for m, _ in sorted_mods[:limit]]


# ----------------------------------------------------------------------
# Module execution
# ----------------------------------------------------------------------
def get_module_score(
    module_name: str, grid: np.ndarray, target: Optional[int] = None, **kwargs
) -> np.ndarray:
    """Retrieve and execute a specific scoring module from the registry."""
    effective_request_id = kwargs.get("request_id", "N/A")
    if module_name not in REGISTERED_MODULES_BRAIN:
        logger.error(
            f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.",
            extra={"request_id": effective_request_id},
        )
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)
    module_func = REGISTERED_MODULES_BRAIN[module_name]
    kwargs["target"] = target
    score_grid = safe_call(module_func, grid, **kwargs)
    if module_name not in _seen_modules_once:
        logger.info(
            f"Executing module FIRST time: {module_name}",
            extra={"request_id": effective_request_id},
        )
        _seen_modules_once.add(module_name)
    else:
        logger.debug(
            f"Executing module: {module_name}",
            extra={"request_id": effective_request_id},
        )
    if isinstance(score_grid, (list, tuple)):
        if score_grid and isinstance(score_grid[0], tuple):
            score_grid = [_to_native_coord(p) for p in score_grid]
        elif len(score_grid) == 2 and all(
            isinstance(x, (int, np.integer)) for x in score_grid
        ):
            score_grid = _to_native_coord(score_grid)
    try:
        return score_grid
    except Exception as e:
        logger.error(
            f"Error executing module {module_name}: {e}",
            extra={"request_id": effective_request_id},
        )
        rows, cols = grid.shape
        return np.zeros((rows, cols), dtype=float)


def aggregate_scores(
    stack: np.ndarray, weights: np.ndarray, names: Optional[list[str]] | None = None
) -> np.ndarray:
    """Normalize score maps then combine via weighted sum."""
    mu = stack.mean(axis=(1, 2), keepdims=True)
    sigma = stack.std(axis=(1, 2), keepdims=True) + 1e-6
    stack_z = (stack - mu) / sigma
    weights_normalized = weights / (weights.sum() + 1e-10)
    final = np.tensordot(weights_normalized, stack_z, axes=(0, 0))
    if names:
        contrib = (weights_normalized[:, None, None] * stack_z).mean(axis=(1, 2))
        for n, w, c in zip(names, weights_normalized, contrib):
            logger.info("aggregate | %s | w=%.3f | contrib=%.3f", n, w, float(c))
    return final
