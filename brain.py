import inspect
import logging
import math
import os
import random
from collections import Counter, defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import irfftn, rfftn
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2

from modules import global_offset_cooccurrence, neighbor_value_distribution


def batchable(fn: Callable) -> Callable:
    """Decorator to allow modules to accept batch or single board input."""

    @wraps(fn)
    def wrapper(boards: np.ndarray, *args, **kwargs) -> np.ndarray:
        boards = np.asarray(boards)
        if boards.ndim == 2:
            return fn(boards, *args, **kwargs)
        return np.stack([fn(b, *args, **kwargs) for b in boards], axis=0)

    return wrapper


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
                                for l in range(k + 1, n):
                                    if line[l] == -1:
                                        gap_cnt += 1
                                        if gap_cnt > allow_gaps:
                                            break
                                        continue
                                    expected = seq_vals[-1] + diff
                                    if math.isclose(line[l], expected, rel_tol=1e-9):
                                        seq_vals.append(line[l])
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
    """Return sliding-window histogram (vectorised via FFT)."""
    rows, cols = grid.shape
    one_hot = np.eye(bins, dtype=float)[grid]  # (r,c,b)
    kernel = np.ones((win, win), dtype=float)
    # FFT-based convolution per bin
    k_fft = rfftn(kernel, s=grid.shape)
    convs = []
    for b in range(bins):
        x_fft = rfftn(one_hot[..., b], s=grid.shape)
        convs.append(irfftn(x_fft * k_fft, s=grid.shape))
    hist = np.stack(convs, axis=-1)
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
    eq_h = (grid[:, :-1] == grid[:, 1:]).astype(float)
    eq_v = (grid[:-1, :] == grid[1:, :]).astype(float)
    score = np.zeros_like(grid, dtype=float)
    score[:, :-1] += eq_h
    score[:, 1:] += eq_h
    score[:-1, :] += eq_v
    score[1:, :] += eq_v
    cross = ndi.generic_filter(
        grid, lambda x: np.sum(x[1:] == x[0]), size=3, mode="constant", cval=-1
    )
    score += cross / 4.0
    score /= score.max(initial=1)
    return score


@batchable
def EXT_Q6_LineBridge_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    return compute_line_bridge_score(grid)


def compute_local_variance_prior(grid, w=3):
    g = grid.astype(float)
    m = ndi.uniform_filter(g, size=w, mode="reflect")
    v = ndi.uniform_filter(g**2, size=w, mode="reflect") - m**2
    inv = 1 / (v + 1e-6)
    inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
    return inv


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
            if len(neighbors) < 2:
                continue
            mean_val = np.mean(neighbors)
            std_val = np.std(neighbors, ddof=1) or 1.0
            row_seq = utils.check_sequences(
                grid[max(0, r - 2) : min(rows, r + 3)], grid, min_len=3, allow_gaps=1
            )
            col_seq = utils.check_sequences(
                grid[:, max(0, c - 2) : min(cols, c + 3)].T,
                grid,
                min_len=3,
                allow_gaps=1,
            )
            legal_values = utils.get_legal_values_for_placement(grid)
            max_score = 0.0
            for val in legal_values:
                deviation = abs(val - mean_val) / std_val
                seq_bonus = (
                    0.3
                    if (row_seq or col_seq) and abs(val - mean_val) > std_val
                    else 0.0
                )
                score = MathUtils().normalize_value(
                    deviation + seq_bonus, 0, max(1.0, std_val + 0.3)
                )
                max_score = max(max_score, score)
            scores[r, c] = max_score
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
    """Composite scoring for proximity and entropy."""
    a2 = get_module_score("EXT_M3_Local_Focus_Vec", grid, request_id=request_id)
    m1 = get_module_score("EXT_M1_Tail_Pattern_Vec", grid, request_id=request_id)
    return 0.65 * a2 + 0.35 * m1


@batchable
def EXT_Q2_PotentialPath_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Composite scoring for potential paths and sequences."""
    m10 = get_module_score("EXT_M10_Sequence_Block_Vec", grid, request_id=request_id)
    f7 = get_module_score("EXT_F7_Strong_Pattern_Vec", grid, request_id=request_id)
    return 0.5 * m10 + 0.5 * f7


@batchable
def EXT_Q3_DiscontinuitySym_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Composite scoring for discontinuity and symmetry."""
    gm20 = get_module_score(
        "EXT_GM20_Skip_Pattern_Confidence_Vec", grid, request_id=request_id
    )
    rows, cols = grid.shape
    sym_score = (
        0.3
        if (
            rows > 1
            and cols > 1
            and grid[rows - 1, cols - 1] == grid[0, 0]
            and grid[rows - 1, cols - 1] != -1
        )
        else 0.0
    )
    return 0.7 * gm20 + sym_score


@batchable
def EXT_Q4_ControlComposite_Vec(
    grid: np.ndarray, request_id: Optional[str] = "N/A"
) -> np.ndarray:
    """Composite scoring for control and error correction."""
    r3 = get_module_score("EXT_R3_Error_Correction_Vec", grid, request_id=request_id)
    other_modules = [
        m
        for m in REGISTERED_MODULES_BRAIN
        if m not in ["EXT_R3_Error_Correction_Vec", "EXT_Q4_ControlComposite_Vec"]
    ]
    mean_score = np.mean(
        [get_module_score(m, grid, request_id=request_id) for m in other_modules],
        axis=0,
    )
    return 0.5 * r3 + 0.5 * mean_score


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
    "EXT_M1_Tail_Pattern_Vec": EXT_M1_Tail_Pattern_Vec,
    "EXT_M3_Local_Focus_Vec": EXT_M3_Local_Focus_Vec,
    "EXT_M10_Sequence_Block_Vec": EXT_M10_Sequence_Block_Vec,
    "EXT_R3_Error_Correction_Vec": EXT_R3_Error_Correction_Vec,
    "EXT_F7_Strong_Pattern_Vec": EXT_F7_Strong_Pattern_Vec,
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
]

RERANK_PHASE = [
    "EXT_Q3_DiscontinuitySym_Vec",
    "EXT_Q6_LineBridge_Vec",
    "EXT_Q7_VariancePrior_Vec",
    "EXT_Q9_MultiScaleEntropy_Vec",
    "EXT_Q10_DistPotential_Vec",
]


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
    if target is not None:
        try:
            sig = inspect.signature(inspect.unwrap(module_func))
            if "target" in sig.parameters:
                kwargs["target"] = target
        except (TypeError, ValueError):
            pass
    score_grid = module_func(grid, **kwargs)
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
