# isort: skip_file
import base64
import heapq
import orjson
import logging
import math
import os
import sys
import re
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from numba import njit

import brain

# fmt: off
# isort: off
from brain import (
    REGISTERED_MODULES_BRAIN,
    BoardAnalyzerUtils,
    aggregate_scores,
    get_module_score,
)
# fmt: on
# isort: on
from modules import FORMULA_REGISTRY, generate_unique_grid
from weights import AGG_WEIGHTS
from neighbor_stats import compute_neighbor_distribution, neighbor_compatibility_score
from csp_solver import heuristic_csp_sampling

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

analyzer_utils = BoardAnalyzerUtils()

# Global position prior cache
_POS_FREQ_CACHE: Dict[str, np.ndarray] = {}


def load_global_pos_freq(samples_dir: str) -> None:
    """Load global position frequency tensor from samples_dir if available."""
    path = Path(samples_dir) / "pos_freq.npz"
    try:
        if path.exists():
            arr = np.load(path)["freq"]
            _POS_FREQ_CACHE[str(path)] = arr.astype(float)
            logger.info("Loaded global position freq from %s", path)
    except Exception as exc:  # pragma: no cover - corrupted file
        logger.error("failed to load %s: %s", path, exc)


def _get_global_pos_freq(samples_dir: str) -> Optional[np.ndarray]:
    path = Path(samples_dir) / "pos_freq.npz"
    key = str(path)
    if key not in _POS_FREQ_CACHE and path.exists():
        load_global_pos_freq(samples_dir)
    return _POS_FREQ_CACHE.get(key)


# 來自 probmap_key_patch_v2.txt
def _native_coord(k):
    return int(k[0]), int(k[1])


def _native_dict(d):
    return {_native_coord(k): v for k, v in d.items()}


def apply_uniqueness_penalty(
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
    strength: float = 0.5,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Penalize cells used by multiple numbers to encourage variety."""

    if strength <= 0:
        return prob_map

    result: Dict[Tuple[int, int], Dict[int, float]] = {}
    for cell, dist in prob_map.items():
        count = len(dist)
        if count > 1:
            factor = 1.0 / (1.0 + strength * (count - 1))
            result[cell] = {n: p * factor for n, p in dist.items()}
        else:
            result[cell] = dist.copy()
    return result


def _iter_json_from_zip(zip_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a zip file with basic validation."""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as f:
                    data = orjson.loads(f.read())
            except Exception as exc:  # pragma: no cover - corrupted JSON
                logger.error("Failed to read %s:%s: %s", zip_path.name, name, exc)
                continue

            if "grid" in data:
                grid = data.get("grid")
                if not isinstance(grid, list) or not all(
                    isinstance(row, list) for row in grid
                ):
                    logger.warning("Invalid grid in %s:%s", zip_path.name, name)
                    continue
                rows = data.get("rows", len(grid))
                cols = data.get("cols", len(grid[0]) if grid else 0)
                if rows != len(grid) or any(len(r) != cols for r in grid):
                    logger.warning("Row/col mismatch in %s:%s", zip_path.name, name)
                    continue
                count += 1
                item = {
                    "rows": rows,
                    "cols": cols,
                    "grid": grid,
                    "target_num": data.get("target_num"),
                    "mode": data.get("mode"),
                }
                yield item
                continue

            for key, boards_list in data.items():
                match = re.match(r"^(\d+)x(\d+)$", key)
                if not match:
                    logger.warning(
                        "Skip invalid key %s in %s:%s", key, zip_path.name, name
                    )
                    continue
                rows, cols = int(match.group(1)), int(match.group(2))
                if not isinstance(boards_list, list):
                    logger.warning(
                        "Invalid boards list for %s in %s:%s", key, zip_path.name, name
                    )
                    continue
                for board in boards_list:
                    if not (
                        isinstance(board, list)
                        and len(board) == rows
                        and all(isinstance(r, list) and len(r) == cols for r in board)
                    ):
                        logger.warning(
                            "Invalid board for %s in %s:%s", key, zip_path.name, name
                        )
                        continue
                    count += 1
                    yield {"rows": rows, "cols": cols, "grid": board}
    logger.info("Loaded %s (%d JSON)", zip_path.name, count)


def iter_sample_jsons(samples_dir: str) -> Iterator[Dict[str, Any]]:
    """Iterate through all JSON samples in ``samples_dir``."""
    path = Path(samples_dir)
    for zp in sorted(path.glob("*.zip")):
        try:
            for item in _iter_json_from_zip(zp):
                yield item
        except Exception as exc:  # pragma: no cover - broken zip
            logger.error("Failed to load %s: %s", zp.name, exc)


def compute_history_frequency(
    samples_dir: str, target_num: int, rows: int, cols: int
) -> np.ndarray:
    """Return frequency matrix for ``target_num`` over all samples."""
    freq = np.zeros((rows, cols), dtype=np.int64)
    total = 0
    for sample in iter_sample_jsons(samples_dir):
        if sample["rows"] != rows or sample["cols"] != cols:
            continue
        total += 1
        grid = sample["grid"]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == target_num:
                    freq[r, c] += 1
    logger.info(
        "History frequency for %d×%d target=%d processed %d samples",
        rows,
        cols,
        target_num,
        total,
    )
    return freq


def compute_position_distribution(
    samples_dir: str,
    rows: int,
    cols: int,
    *,
    mode: Optional[str] = None,
) -> Dict[Tuple[int, int], Dict[int, int]]:
    """Return per-cell number frequency counts from history samples."""
    stats: Dict[Tuple[int, int], Dict[int, int]] = {
        (r, c): defaultdict(int) for r in range(rows) for c in range(cols)
    }
    total = 0
    for sample in iter_sample_jsons(samples_dir):
        if sample["rows"] != rows or sample["cols"] != cols:
            continue
        if mode and sample.get("mode") != mode:
            continue
        total += 1
        grid = np.asarray(sample["grid"], dtype=int)
        for r in range(rows):
            for c in range(cols):
                k = int(grid[r, c])
                stats[(r, c)][k] += 1
    logger.info(
        "Position distribution for %d×%d processed %d samples%s",
        rows,
        cols,
        total,
        f" mode={mode}" if mode else "",
    )
    return {k: dict(v) for k, v in stats.items()}


def compute_number_distribution(
    samples_dir: str,
    rows: int,
    cols: int,
    *,
    mode: Optional[str] = None,
) -> Dict[int, Dict[Tuple[int, int], int]]:
    """Return per-number position frequency counts from history samples."""
    stats: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    for sample in iter_sample_jsons(samples_dir):
        if sample["rows"] != rows or sample["cols"] != cols:
            continue
        if mode and sample.get("mode") != mode:
            continue
        total += 1
        grid = np.asarray(sample["grid"], dtype=int)
        for r in range(rows):
            for c in range(cols):
                k = int(grid[r, c])
                stats[k][(r, c)] += 1
    logger.info(
        "Number distribution for %d×%d processed %d samples%s",
        rows,
        cols,
        total,
        f" mode={mode}" if mode else "",
    )
    return {n: dict(pos) for n, pos in stats.items()}


def predict_number(
    grid_with_blank: List[List[int]] | np.ndarray,
    stats: Dict[Tuple[int, int], Dict[int, int]],
) -> List[Tuple[Tuple[int, int], int, float]]:
    """Predict numbers for blanks in ``grid_with_blank`` using precomputed stats."""
    grid = np.asarray(grid_with_blank, dtype=int)
    rows, cols = grid.shape
    used = {int(v) for v in grid.ravel() if v != -1}
    all_nums = set(range(1, rows * cols + 1))
    remain = all_nums - used
    predictions: List[Tuple[Tuple[int, int], int, float]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -1:
                dist = stats.get((r, c), {})
                total = sum(dist.get(n, 0) for n in remain) or 1
                for n in sorted(remain):
                    freq = dist.get(n, 0)
                    score = float(freq) / float(total)
                    predictions.append(((r, c), n, score))
    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions


@lru_cache(maxsize=8)
def compute_position_probabilities(
    samples_dir: str, rows: int, cols: int
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Return per-cell number probabilities from history samples."""
    global_freq = _get_global_pos_freq(samples_dir)
    if global_freq is not None:
        buckets = global_freq.shape[1]
        prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
        for r in range(rows):
            for c in range(cols):
                u = r / (rows - 1) if rows > 1 else 0.0
                v = c / (cols - 1) if cols > 1 else 0.0
                i = min(int(u * buckets), buckets - 1)
                j = min(int(v * buckets), buckets - 1)
                dist = global_freq[:, i, j].astype(float)
                dist = dist[: rows * cols + 1]
                tot = dist.sum() or 1.0
                probs = {k: dist[k] / tot for k in range(1, dist.size) if dist[k] > 0}
                prob_map[(r, c)] = probs
        return prob_map

    cached = Path(samples_dir) / "prior.npy"
    if cached.exists():
        cube = np.load(cached, mmap_mode="r")
        if cube.shape[:2] != (rows, cols):
            logger.warning("Cached prior shape mismatch: %s", cube.shape)
        else:
            logger.info("Loaded prior from %s", cached)
            prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
            for r in range(rows):
                for c in range(cols):
                    dist = cube[r, c].astype(float)
                    total = dist.sum() or 1.0
                    probs = {
                        i: dist[i] / total for i in range(1, dist.size) if dist[i] > 0
                    }
                    prob_map[(r, c)] = probs
            return prob_map

    counts = np.zeros((rows, cols, rows * cols + 1), dtype=np.int64)
    total = 0
    for sample in iter_sample_jsons(samples_dir):
        if sample["rows"] != rows or sample["cols"] != cols:
            continue
        total += 1
        grid = np.asarray(sample["grid"], dtype=int)
        mask = (grid >= 1) & (grid <= rows * cols)
        rr, cc = np.indices(grid.shape)
        np.add.at(counts, (rr[mask], cc[mask], grid[mask]), 1)

    prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for r in range(rows):
        for c in range(cols):
            dist = counts[r, c]
            total_cell = dist.sum()
            if total_cell:
                prob_map[(r, c)] = {
                    n: dist[n] / float(total_cell)
                    for n in range(1, rows * cols + 1)
                    if dist[n] > 0
                }
            else:
                prob_map[(r, c)] = {}

    logger.info(
        "Position frequencies for %d×%d processed %d samples",
        rows,
        cols,
        total,
    )
    return prob_map


@lru_cache(maxsize=8)
def compute_global_distribution(samples_dir: str, rows: int, cols: int) -> np.ndarray:
    """Return per-cell number probability cube from history samples."""
    cached = Path(samples_dir) / "prior.npy"
    if cached.exists():
        cube = np.load(cached, mmap_mode="r")
        if cube.shape[:2] != (rows, cols):
            logger.warning("Cached prior shape mismatch: %s", cube.shape)
        else:
            logger.info("Loaded prior cube from %s", cached)
            totals = cube.sum(axis=2, keepdims=True)
            totals[totals == 0] = 1
            return cube.astype(float) / totals

    counts = np.zeros((rows, cols, rows * cols + 1), dtype=np.int64)
    total = 0
    for sample in iter_sample_jsons(samples_dir):
        if sample["rows"] != rows or sample["cols"] != cols:
            continue
        total += 1
        grid = np.asarray(sample["grid"], dtype=int)
        mask = (grid >= 1) & (grid <= rows * cols)
        rr, cc = np.indices(grid.shape)
        np.add.at(counts, (rr[mask], cc[mask], grid[mask]), 1)

    totals = counts.sum(axis=2, keepdims=True)
    totals[totals == 0] = 1
    probs = counts.astype(float) / totals
    logger.info(
        "Global distribution for %d×%d processed %d samples",
        rows,
        cols,
        total,
    )
    return probs


def adjust_weights_based_on_history(
    history: Dict[str, float], formulas: Tuple[str, ...]
) -> np.ndarray:
    """Dynamically adjust formula weights based on historical performance."""
    total = sum(history.get(f, 0.0) for f in formulas) or 1e-10
    return np.array([history.get(f, 0.0) / total for f in formulas])


def dump_prior(samples_dir: str, outfile: str) -> None:
    """Aggregate all samples into a prior cube and save as ``outfile``."""
    cube = None
    rows = cols = 0
    for sample in iter_sample_jsons(samples_dir):
        r, c = sample["rows"], sample["cols"]
        if cube is None:
            rows, cols = r, c
            cube = np.zeros((rows, cols, rows * cols + 1), dtype=np.int64)
        if r != rows or c != cols:
            logger.warning("Skip mismatched sample %s×%s", r, c)
            continue
        grid = np.asarray(sample["grid"], dtype=int)
        mask = (grid >= 1) & (grid <= rows * cols)
        rr, cc = np.indices(grid.shape)
        np.add.at(cube, (rr[mask], cc[mask], grid[mask]), 1)
    if cube is None:
        logger.error("No valid samples found in %s", samples_dir)
        return
    np.save(outfile, cube)
    logger.info("Prior saved to %s", outfile)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    if len(sys.argv) == 3:
        dump_prior(sys.argv[1], sys.argv[2])


def select_modules(grid: np.ndarray, target: Optional[int] = None) -> List[str]:
    """Select up to ``CORE_LIMIT`` modules based on weights and scores."""
    if os.getenv("FORCE_FULL_SCAN", "0") == "1":
        return list(REGISTERED_MODULES_BRAIN)

    base_modules = brain.get_core_modules()
    scores = {
        m: float(np.mean(get_module_score(m, grid, target=target)))
        for m in base_modules
    }
    return sorted(scores, key=scores.get, reverse=True)


def fill_masked_randomly(
    grid: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Fill cells marked by ``mask`` with a permutation of remaining numbers."""

    g = np.asarray(grid, dtype=int).copy()
    mask_arr = np.asarray(mask, dtype=bool)
    blanks = np.argwhere(mask_arr)
    if blanks.size == 0:
        return g

    rows, cols = g.shape
    all_vals = np.arange(1, rows * cols + 1)
    remain = np.setdiff1d(all_vals, g[~mask_arr], assume_unique=True)
    rng.shuffle(remain)
    g[blanks[:, 0], blanks[:, 1]] = remain[: blanks.shape[0]]
    return g


def generate_full_boards(
    rows: int,
    cols: int,
    batch: int,
    rng: np.random.Generator,
    formulas: Tuple[str, ...],
    weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Generate batch of complete boards using weighted formulas with importance sampling."""
    valid = [f for f in formulas if f in FORMULA_REGISTRY]
    if not valid:
        raise ValueError("No valid formulas available")
    weights = np.array(
        [weights[i] for i, f in enumerate(formulas) if f in FORMULA_REGISTRY],
        dtype=float,
    )
    weights = weights / (weights.sum() + 1e-10)
    boards = np.empty((batch, rows, cols), dtype=np.int16)
    known_mask = grid == -1
    known_vals = grid[~known_mask]
    for i in range(batch):
        choice = rng.choice(valid, p=weights)
        base = FORMULA_REGISTRY[choice](rows, cols, rng)
        board = np.asarray(base, dtype=np.int16)
        board[~known_mask] = known_vals
        boards[i] = board
    return boards


# JIT-accelerated count update for board batches
@njit
def _update_counts(
    counts: np.ndarray,
    board: np.ndarray,
    br: np.ndarray,
    bc: np.ndarray,
    idxs: np.ndarray,
    num_map: np.ndarray,
    weights: np.ndarray,
) -> None:
    for i in range(br.size):
        num = board[br[i], bc[i]]
        idx_num = num_map[num]
        if idx_num >= 0:
            counts[idxs[i], idx_num] += weights[br[i], bc[i]]


@njit
def _update_counts_jit(
    counts: np.ndarray,
    boards: np.ndarray,
    br: np.ndarray,
    bc: np.ndarray,
    idxs: np.ndarray,
    num_map: np.ndarray,
    weights: np.ndarray,
) -> None:
    for b in range(boards.shape[0]):
        board = boards[b]
        for i in range(br.size):
            num = board[br[i], bc[i]]
            idx_num = num_map[num]
            if idx_num >= 0:
                counts[idxs[i], idx_num] += weights[br[i], bc[i]]


def simulate_full_board(
    grid: np.ndarray,
    target_num: Optional[int],
    n_iter: int = 6000,
    rng: Optional[np.random.Generator] = None,
    *,
    focus_cells: Optional[List[Tuple[int, int]]] = None,
    epsilon: float = 0.0,
    threshold: float = 1e-3,
    check_interval: int = 500,
    mask: Optional[np.ndarray] = None,
    _internal: bool = False,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Simulate full boards with optional focus and ε-exploration."""
    logger.info(
        "simulate_full_board called: target_num=%s, n_iter=%d",
        str(target_num),
        n_iter,
    )
    if rng is None:
        rng = np.random.default_rng()

    g = np.asarray(grid, dtype=np.int16)
    rows, cols = g.shape
    mask_arr = np.asarray(mask, dtype=bool) if mask is not None else (g == -1)
    blanks = np.argwhere(mask_arr)
    known = np.argwhere(~mask_arr)
    known_vals = g[~mask_arr]
    grid_gen = g.copy()
    grid_gen[mask_arr] = -1
    legal_all = analyzer_utils.get_legal_values_for_placement(grid_gen)

    if target_num is not None:
        count_map = np.zeros((rows, cols), dtype=int)
        for _ in range(max(1, n_iter)):
            filled = fill_masked_randomly(g, mask_arr, rng)
            mask_hit = filled == target_num
            count_map += mask_hit.astype(int)

        prob_map = {}
        for r, c in blanks:
            prob_map[(int(r), int(c))] = {
                target_num: float(count_map[r, c]) / float(max(1, n_iter))
            }
        return prob_map

    # Enhanced module selection for importance sampling
    modules = select_modules(g, target=target_num)
    module_scores = np.mean(
        [get_module_score(mod, g, target=target_num) for mod in modules],
        axis=0,
    )
    importance_weights = np.where(mask_arr, module_scores, 0).flatten()
    importance_weights = importance_weights / (np.sum(importance_weights) + 1e-10)

    # Dynamic formula weights based on grid pattern
    history = {"random_entropy": 0.4, "shuffle": 0.3, "tail_cluster": 0.3}
    if np.mean(module_scores) > 0.6:
        history["tail_cluster"] += 0.1
        history["random_entropy"] -= 0.05

    formulas = ("random_entropy", "shuffle", "tail_cluster")
    weights = adjust_weights_based_on_history(history, formulas)
    remain = n_iter
    counts = np.zeros((blanks.shape[0], rows * cols + 1), dtype=float)
    num_map = -np.ones(rows * cols + 1, dtype=np.int32)
    for j, n in enumerate(range(1, rows * cols + 1)):
        num_map[n] = j
    br, bc = blanks[:, 0].astype(np.int32), blanks[:, 1].astype(np.int32)
    blank_index = {(int(r), int(c)): i for i, (r, c) in enumerate(blanks)}
    prev_probs = np.zeros_like(counts)
    batch_progress = 0
    focus_set = {tuple(fc) for fc in focus_cells} if focus_cells else None
    other_cells = (
        [tuple(b) for b in blanks if tuple(b) not in focus_set] if focus_set else []
    )
    early_stop = False

    while remain > 0:
        batch = min(4000, remain)
        boards = generate_full_boards(
            rows, cols, batch, rng, formulas, weights, grid_gen
        )

        if known.size:
            mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
            boards = boards[mask]
            if len(boards) == 0:
                batch = min(batch * 2, 8000)
                boards = generate_full_boards(
                    rows, cols, batch, rng, formulas, weights, grid_gen
                )
                mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
                boards = boards[mask]

        if len(boards) > 0:
            for board in boards:
                mods_fast = ["focus", "skip", "diff"]
                stack_fast = np.stack(
                    [get_module_score(m, board) for m in mods_fast], axis=0
                )
                w_fast = np.array([AGG_WEIGHTS[m] for m in mods_fast])
                fast = aggregate_scores(stack_fast, w_fast, mods_fast)
                tau_local = float(os.getenv("TAU_SOFTMAX", "1.0"))
                soft_fast = np.exp(fast / tau_local)
                soft_fast /= soft_fast.sum() + 1e-10
                fast = soft_fast
                cells_iter = blanks if focus_set is None else focus_set
                if focus_set is not None and other_cells and rng.random() < epsilon:
                    cells_iter = list(focus_set) + [
                        other_cells[int(rng.integers(len(other_cells)))]
                    ]
                weights_cell = fast
                indices = np.array(
                    [blank_index[(int(r), int(c))] for r, c in cells_iter],
                    dtype=np.int32,
                )
                br_sel = br[indices]
                bc_sel = bc[indices]
                idxc_all = br_sel * cols + bc_sel
                mask_sel = rng.random(indices.size) < importance_weights[idxc_all]
                if mask_sel.any():
                    _update_counts_jit(
                        counts,
                        board[np.newaxis, :],
                        br_sel[mask_sel],
                        bc_sel[mask_sel],
                        indices[mask_sel],
                        num_map,
                        weights_cell,
                    )
        remain -= batch
        batch_progress += batch
        if batch_progress >= check_interval:
            totals = counts.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            probs = counts / totals
            delta = np.abs(probs - prev_probs).sum()
            if delta < threshold:
                early_stop = True
                break
            prev_probs = probs.copy()
            batch_progress = 0

    totals = counts.sum(axis=1)
    prob_map = {}
    for idx, (r, c) in enumerate(blanks):
        total = totals[idx] or 1e-10
        cell: Dict[int, float] = {}
        for n in legal_all:
            j = num_map[n]
            if j >= 0:
                cell[n] = max(counts[idx, j] / total, 1e-10)

        prob_map[(int(r), int(c))] = cell

    if early_stop and not _internal and blanks.size > 0:
        change = np.abs((counts / (totals[:, None] + 1e-10)) - prev_probs).sum(axis=1)
        m = max(1, int(0.2 * blanks.shape[0]))
        idx_top = np.argsort(change)[-m:]
        focus = [tuple(map(int, blanks[i])) for i in idx_top]
        refine = simulate_full_board(
            g,
            target_num,
            n_iter=max(1, int(n_iter * 0.2)),
            rng=rng,
            focus_cells=focus,
            epsilon=epsilon,
            threshold=threshold,
            check_interval=check_interval,
            mask=mask_arr,
            _internal=True,
        )
        for cell in focus:
            prob_map[cell] = refine.get(cell, prob_map.get(cell, {}))

    # Two-phase scoring: Re-rank top K candidates with Borda or Soft-Max
    tau = float(os.getenv("TAU_SOFTMAX", "0.3"))
    mods_rerank = ["focus", "skip", "diff", "mirror", "conn", "tail"]
    w = np.array([AGG_WEIGHTS[m] for m in mods_rerank])
    stack_rerank = np.stack(
        [get_module_score(m, g, target=target_num) for m in mods_rerank],
        axis=0,
    )
    final_heat = aggregate_scores(stack_rerank, w, mods_rerank)
    soft_heat = np.exp(final_heat / tau)
    soft_heat /= soft_heat.sum() + 1e-10
    topk = int(os.getenv("TOPK_RERANK", "100"))
    if topk < 0:  # -1 表示 rows × cols
        topk = rows * cols

    candidates = [
        (r, c, max(probs.values()), num)
        for (r, c), probs in prob_map.items()
        for num in probs
    ]
    top_k = heapq.nlargest(topk, candidates, key=lambda x: x[2])
    final_prob_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for r, c, _, num in top_k:
        base = prob_map[(r, c)].get(num, 0.0)
        final_score = float(soft_heat[r, c]) * base
        cell = final_prob_map.setdefault((r, c), {})
        cell[num] = final_score

    for cell, dist in final_prob_map.items():
        total = sum(dist.values()) or 1e-10
        for n in dist:
            dist[n] /= total

    # 來自 probmap_key_patch_v2.txt
    prob_map = {(int(r), int(c)): cell for (r, c), cell in final_prob_map.items()}

    prob_map = apply_uniqueness_penalty(prob_map)

    # --- 保證所有格都有 entry ------------------------------
    if os.getenv("FORCE_FULL_SCAN", "0") == "1":
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in prob_map:
                    prob_map[(r, c)] = {n: 0.0 for n in range(100)}
    # --------------------------------------------------------

    return prob_map


def weight_prob_by_modules(
    grid: np.ndarray,
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
    target_num: Optional[int] = None,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    if not isinstance(prob_map, dict):
        logger.error(f"Invalid prob_map type: {type(prob_map)}")
        return {}

    result = prob_map.copy()
    modules = select_modules(grid, target=target_num)
    module_scores = Parallel(n_jobs=4)(
        delayed(get_module_score)(mod, grid, target=target_num) for mod in modules
    )
    module_scores = np.array(module_scores)

    for (r, c), probs in result.items():
        if (r, c) not in prob_map:
            continue
        scores = module_scores[:, r, c]
        scores = np.nan_to_num(scores, nan=0.0)
        softmax_scores = np.exp(scores / 0.5)
        softmax_scores /= softmax_scores.sum() + 1e-10
        scale = float(np.linalg.norm(softmax_scores, ord=2))

        if target_num is not None:
            if target_num in probs:
                probs[target_num] = max(probs[target_num] * scale, 1e-10)
                total = probs[target_num] or 1e-10
                result[(r, c)] = {target_num: probs[target_num] / total}
            else:
                result[(r, c)] = {target_num: 0.0}
        else:
            for val in probs:
                probs[val] = max(probs[val] * scale, 1e-10)
            total = sum(probs.values()) or 1e-10
            result[(r, c)] = {k: v / total for k, v in probs.items()}

    return _native_dict(result)


def _compute_final_recommendations(
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
    module_norm: np.ndarray,
    target_num: Optional[int],
    fusion_alpha: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Return fused ranking of cells based on probabilities and module scores."""
    recs: List[Dict[str, Any]] = []
    for (r, c), dist in prob_map.items():
        if target_num is not None:
            prob = dist.get(target_num, 0.0)
        else:
            prob = max(dist.values()) if dist else 0.0
        score = fusion_alpha * float(module_norm[r, c]) + (1.0 - fusion_alpha) * float(
            prob
        )
        recs.append({"row": int(r), "col": int(c), "score": score})
    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[:top_k]


def fuse_predictions_with_heatmap(
    heatmap: np.ndarray,
    predictions: List[Dict[str, Any]],
    *,
    fusion_alpha: float = 0.7,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Fuse prediction scores with heatmap probability.

    Parameters
    ----------
    heatmap : np.ndarray
        Probability matrix for the target number.
    predictions : list of dict
        Items with ``row``, ``col`` and ``score`` fields (0-based indices).
    fusion_alpha : float, optional
        Weight for ``score`` from predictions. ``1 - fusion_alpha`` is applied
        to ``heatmap`` probability. Defaults to ``0.7``.
    top_k : int, optional
        Number of cells to return. Defaults to ``3``.

    Returns
    -------
    List[Dict[str, Any]]
        Sorted list of recommendations with ``final_score``.
    """

    rows, cols = heatmap.shape
    score_map = {
        (int(p.get("row")), int(p.get("col"))): float(p.get("score", 0.0))
        for p in predictions
    }
    recs: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            pred_score = score_map.get((r, c), 0.0)
            prob = float(heatmap[r, c])
            final = fusion_alpha * pred_score + (1.0 - fusion_alpha) * prob
            recs.append({"row": r, "col": c, "final_score": final})
    recs.sort(key=lambda x: x["final_score"], reverse=True)
    return recs[:top_k]


def fuse_score_matrices(
    predict_scores: np.ndarray,
    heatmap_prob_map: np.ndarray,
    *,
    fusion_alpha: float = 0.5,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Fuse two full-size score matrices and return top ranked cells."""

    if predict_scores.shape != heatmap_prob_map.shape:
        raise ValueError("score map and heatmap must have the same shape")

    rows, cols = predict_scores.shape
    recs: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            pred = float(predict_scores[r, c])
            prob = float(heatmap_prob_map[r, c])
            score = fusion_alpha * pred + (1.0 - fusion_alpha) * prob
            recs.append({"row": r, "col": c, "final_score": score})

    recs.sort(key=lambda x: x["final_score"], reverse=True)
    return recs[:top_k]


def rank_cells_by_prior_and_modules(
    grid: np.ndarray,
    prior_cube: np.ndarray,
    modules: List[str],
    module_weights: Optional[List[float]] = None,
    *,
    target_num: int,
    w_prior: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Return top-3 unknown cells ranked by fused prior and module scores."""

    rows, cols = grid.shape
    if prior_cube.shape[:2] != (rows, cols):
        raise ValueError("prior cube shape mismatch")

    if module_weights is None:
        module_weights = [1.0] * len(modules)
    if len(module_weights) != len(modules):
        raise ValueError("weights length mismatch")

    agg = np.zeros((rows, cols), dtype=float)
    for mod, w in zip(modules, module_weights):
        agg += w * get_module_score(mod, grid, target=target_num)

    prior_k = (
        prior_cube[:, :, target_num]
        if prior_cube.shape[2] > target_num
        else np.zeros((rows, cols), dtype=float)
    )
    final = w_prior * prior_k + (1.0 - w_prior) * agg

    mask = grid == -1
    total = final[mask].sum() or 1.0
    final[mask] = final[mask] / total

    results = [
        (int(r), int(c), float(final[r, c] * 100.0))
        for r in range(rows)
        for c in range(cols)
        if mask[r, c]
    ]
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:3]


def assign_unique_numbers(
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
) -> Dict[int, Tuple[int, int]]:
    """Assign each number to a unique cell maximizing overall probability."""
    try:
        from scipy.optimize import linear_sum_assignment

        cells = list(prob_map.keys())
        nums = sorted({n for d in prob_map.values() for n in d})
        cost = np.full((len(nums), len(cells)), 50.0, dtype=float)

        for i, num in enumerate(nums):
            for j, cell in enumerate(cells):
                prob = max(prob_map[cell].get(num, 1e-10), 1e-10)
                cost[i, j] = -math.log(prob)

        row, col = linear_sum_assignment(cost)
        return {nums[r]: cells[c] for r, c in zip(row, col)}
    except Exception as e:  # pragma: no cover - fallback rarely used
        logger.error("assign_unique_numbers failed: %s", e)
        assigned: Dict[int, Tuple[int, int]] = {}
        used: set[Tuple[int, int]] = set()
        numbers = sorted({n for d in prob_map.values() for n in d})
        for num in numbers:
            best_cell = None
            best_p = -1.0
            for cell, dist in prob_map.items():
                if cell in used:
                    continue
                p = dist.get(num, 0.0)
                if p > best_p:
                    best_p = p
                    best_cell = cell
            if best_cell is not None:
                assigned[num] = best_cell
                used.add(best_cell)
        return assigned


def global_unique(
    prob_map: Dict[Tuple[int, int], Dict[int, float]],
    blanks: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Tuple[int, float]]:
    try:
        assignments = assign_unique_numbers(prob_map)
        return {
            cell: (num, prob_map[cell].get(num, 0.0))
            for num, cell in assignments.items()
        }
    except Exception as e:
        logger.error(f"Global unique assignment failed: {e}")
        assigned, res = set(), {}
        for cell in sorted(
            blanks,
            key=lambda p: max(prob_map[p].values() or [0]),
            reverse=True,
        ):
            for n, p in sorted(
                prob_map[cell].items(), key=lambda x: x[1], reverse=True
            ):
                if n not in assigned:
                    assigned.add(n)
                    res[cell] = (n, p)
                    break
            if cell not in res:
                res[cell] = (
                    (list(prob_map[cell].keys())[0], 0.0)
                    if prob_map[cell]
                    else (1, 0.0)
                )
        return res


class MCTSNode:
    EPS = 1e-9  # 檔頭或 class 內自訂常數

    def __init__(self, grid, parent=None, parent_action=None):
        self.grid = grid.copy()
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.virtual_loss = 0
        self.untried_actions = [
            (r, c, v)
            for r, c in np.argwhere(grid == -1)
            for v in analyzer_utils.get_legal_values_for_placement(grid)
        ]

    def uct_select(self):
        """Upper-Confidence bound with virtual-loss safe division"""

        def ucb(child):
            denom = child.visits + child.virtual_loss
            if denom == 0:
                return float("inf")  # 確保新節點優先被選
            exploitation = child.value / denom
            exploration = math.sqrt(2 * math.log(self.visits + 1) / denom)
            return exploitation + exploration

        return max(self.children, key=ucb)


def mcts(grid: np.ndarray, iterations: int = 1000):
    rows, cols = grid.shape
    root = MCTSNode(grid)

    def simulate(node):
        try:
            current = node
            while (
                current.untried_actions
                and len(current.children) < 1.5 * current.visits**0.5
            ):
                current = current.uct_select()
                current.virtual_loss += 1
            if current.untried_actions:
                r, c, v = current.untried_actions.pop()
                new_grid = current.grid.copy()
                new_grid[r, c] = v
                new_child = MCTSNode(new_grid, current, (r, c, v))
                new_child.visits = (
                    1  # 或 new_child.visits = new_child.virtual_loss = EPS
                )
                current.children.append(new_child)
                current = new_child

            sim_result = simulate_full_board(current.grid, None, n_iter=100)
            if not isinstance(sim_result, dict):
                logger.error(f"Invalid sim_result type: {type(sim_result)}")
                return 0.0

            reward = 0.0
            for r, c in np.argwhere(grid == -1):
                if (r, c) in sim_result:
                    weighted = weight_prob_by_modules(
                        current.grid, {(r, c): sim_result[(r, c)]}
                    )
                    reward += max(weighted[(r, c)].values())

            while current is not None:
                current.visits += 1
                current.value += reward
                current.virtual_loss -= 1
                current = current.parent
            return reward
        except ZeroDivisionError as e:
            logger.error(f"ZeroDivisionError in simulate: {e}")
            return 0.0

    Parallel(n_jobs=4, require="sharedmem")(
        delayed(simulate)(root) for _ in range(iterations // 4)
    )
    best_child = max(root.children, key=lambda c: c.value / c.visits, default=root)
    return best_child.grid


# Main prediction entry point
def predict_scratch_card(
    grid: List[List[int]],
    target_num: Optional[int] = None,
    iterations: Optional[int] = None,
    quick_iter: Optional[int] = None,
    refine_iter: Optional[int] = None,
    min_total_iter: Optional[int] = None,
    unique: bool = True,
    *,
    global_iter: Optional[int] = None,
    focus_iter: Optional[int] = None,
    top_n: int = 10,
    epsilon: float = 0.05,
    result_top_k: Optional[int] = None,
    priors: Optional[Dict[int, float]] = None,
    history_dir: str = "samples",
    gamma_history: float = 0.0,
    sample_gamma: float = 0.0,
    fusion_alpha: float = 0.5,
    pseudo_count: float = 0.0,
    exclude_filled: bool = True,
    strategy: str = "legacy",
) -> Dict[str, Any]:

    BLANK_VAL = -1
    # Keep object dtype first to avoid coercing values like "0" or "" to 0
    grid_np = np.asarray(grid, dtype=object)
    grid_np = np.where(grid_np == BLANK_VAL, BLANK_VAL, grid_np).astype(np.int64)
    rows, cols = grid_np.shape

    blanks = [tuple(b) for b in np.argwhere(grid_np == BLANK_VAL)]

    known_coords = [tuple(b) for b in np.argwhere(grid_np > 0)]
    use_heatmap_only = False
    if len(known_coords) <= 3:
        has_adj = False
        for r, c in known_coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid_np[nr, nc] > 0:
                        has_adj = True
                        break
                if has_adj:
                    break
        if not has_adj:
            use_heatmap_only = True

    if not blanks:
        return {
            "mode": "no_blanks",
            "strategy": "predict_structured",
            "predictions": [],
            "top_predictions": [],
            "full_probabilities": {},
            "final_recommendations": [],
        }

    if use_heatmap_only and target_num is not None:
        heat = probability_heatmap(
            grid_np,
            target_num,
            n_iter=iterations or 10000,
            sample_gamma=sample_gamma,
            history_dir=history_dir,
        )
        top_k = result_top_k or int(os.getenv("RESULT_TOP_K", "3"))
        ranked = sorted([(float(heat[r, c]), r, c) for r, c in blanks], reverse=True)[
            :top_k
        ]
        preds = [{"row": r, "col": c, "score": prob} for prob, r, c in ranked]
        return {
            "mode": "heatmap_only",
            "strategy": "heatmap_only",
            "target_num": int(target_num),
            "predictions": preds,
            "top_predictions": preds,
            "full_probabilities": {},
            "final_recommendations": [],
        }

    dist = compute_neighbor_distribution(rows, cols, target_num, n_sims=10000)
    nbr_score = neighbor_compatibility_score(grid_np, dist)

    nbr_probs = {(int(r), int(c)): float(nbr_score[r, c]) for r, c in blanks}
    csp_probs = heuristic_csp_sampling(
        grid_np.tolist(),
        target_num,
        nbr_probs,
        samples=iterations or 2000,
        enforce_rowcol=True,
    )

    scores_uniform = False
    neighbor_counts = []
    if blanks:
        scores = np.array([nbr_score[pos] for pos in blanks], dtype=float)
        scores_uniform = np.ptp(scores) < epsilon
        for r, c in blanks:
            cnt = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and grid_np[nr, nc] != BLANK_VAL
                    ):
                        cnt += 1
            neighbor_counts.append(cnt)

    final_score_map = np.zeros_like(grid_np, dtype=float)
    if target_num is not None and (
        scores_uniform or (neighbor_counts and max(neighbor_counts) <= 1)
    ):
        heat = probability_heatmap(
            grid_np,
            target_num,
            n_iter=iterations or 500,
            sample_gamma=sample_gamma,
            history_dir=history_dir,
        )
        for r, c in blanks:
            final_score_map[r, c] = float(heat[r, c])
    else:
        alpha = 0.7
        for r, c in blanks:
            final_score_map[r, c] += alpha * nbr_probs.get((r, c), 0.0)
        for (r, c), p in csp_probs.items():
            final_score_map[r, c] += (1 - alpha) * p

    top_k = result_top_k or int(os.getenv("RESULT_TOP_K", "3"))
    rings = BoardAnalyzerUtils.ring_index(*grid_np.shape)
    ranked_all = sorted(
        blanks,
        key=lambda pos: (rings[pos], -final_score_map[pos]),
    )
    ranked = ranked_all[:top_k]

    preds = [
        {"row": r, "col": c, "score": float(final_score_map[r, c])} for r, c in ranked
    ]

    return {
        "mode": "neighbor",
        "strategy": "predict_structured",
        "predictions": preds,
        "top_predictions": preds,
        "full_probabilities": {},
        "final_recommendations": [],
    }


def process_grid(grid):
    blanks = np.argwhere(grid == -1)
    preds = []
    for r, c in blanks:
        legal_vals = analyzer_utils.get_legal_values_for_placement(grid)
        max_prob = max(legal_vals) if legal_vals else 1
        preds.append(
            {
                "row": int(r),
                "col": int(c),
                "candidates": [int(max_prob)],
                "probability": 100.0 if grid[r, c] != -1 else 50.0,
            }
        )
    return preds


def monte_carlo_prob_map(
    grid: Union[List[List[int]], np.ndarray],
    k: Optional[int],
    n_iter: int = 1000,
    *,
    seed: int = 0,
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Estimate number distribution via Monte-Carlo sampling.

    Parameters
    ----------
    grid : List[List[int]] or np.ndarray
        Board matrix where ``-1`` denotes an unknown cell.
    k : int or None
        Target number to estimate. ``None`` computes probability for all
        remaining numbers.
    n_iter : int
        Simulation iterations.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray or Dict[int, np.ndarray]
        If ``k`` is provided, a 2-D probability matrix of the same shape as the
        grid is returned. Otherwise a mapping from number to probability matrix
        is produced.
    """

    g = np.asarray(grid, dtype=int)
    rng = np.random.default_rng(seed)

    rows, cols = g.shape
    blanks = np.argwhere(g == -1)
    blank_idx = (blanks[:, 0], blanks[:, 1])
    known_vals = g[g != -1]
    all_vals = np.arange(1, rows * cols + 1)
    remain = np.setdiff1d(all_vals, known_vals, assume_unique=True)

    if k is not None and k not in all_vals:
        raise ValueError("k out of range")

    if k is not None:
        counts = np.zeros((rows, cols), dtype=int)
        prev = np.zeros_like(counts, dtype=float)
    else:
        counts = {int(val): np.zeros((rows, cols), dtype=int) for val in remain}
        prev = {int(val): np.zeros((rows, cols), dtype=float) for val in remain}

    actual_iter = 0
    for it in range(1, max(1, n_iter) + 1):
        sample = rng.permutation(remain)
        board = g.copy()
        board[blank_idx] = sample[: blanks.shape[0]]

        if k is not None:
            hits = board == k
            counts += hits.astype(int)
            if it % 500 == 0:
                curr = counts.astype(float) / float(it)
                delta = np.abs(curr - prev).sum()
                if delta < 1e-3:
                    break
                prev = curr.copy()
        else:
            for val in remain:
                counts[int(val)] += (board == val).astype(int)
            if it % 500 == 0:
                converged = True
                for val in remain:
                    curr = counts[int(val)].astype(float) / float(it)
                    delta = np.abs(curr - prev[int(val)]).sum()
                    if delta >= 1e-3:
                        converged = False
                    prev[int(val)] = curr
                if converged:
                    break
        actual_iter = it

    if k is not None:
        if actual_iter == 0:
            actual_iter = n_iter
        prob = counts.astype(float) / float(actual_iter)
        prob[g != -1] = 0.0
        return prob

    prob_all: Dict[int, np.ndarray] = {}
    for val, mat in counts.items():
        iters = actual_iter if actual_iter else n_iter
        arr = mat.astype(float) / float(iters)
        arr[g != -1] = 0.0
        prob_all[int(val)] = arr
    return prob_all


def prob_map_to_png(prob_map: np.ndarray) -> bytes:
    """Render probability matrix to a grayscale PNG."""
    import struct
    import zlib

    h, w = prob_map.shape
    img = np.clip(prob_map, 0.0, 1.0)
    img8 = (img * 255).astype(np.uint8)

    raw = b"".join(b"\x00" + img8[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return png


def heatmap_to_base64(prob_map: np.ndarray) -> str:
    """Convert probability map to a base64-encoded PNG."""
    png_bytes = prob_map_to_png(prob_map)
    return base64.b64encode(png_bytes).decode("ascii")


def render_heatmap(prob_map: np.ndarray, output_format: str = "base64") -> Any:
    """Return heatmap in the desired format.

    Parameters
    ----------
    prob_map : np.ndarray
        Probability matrix to render.
    output_format : str
        One of ``"raw"``, ``"base64"``, or ``"png_bytes"``.
    """

    fmt = output_format.lower()
    if fmt == "raw":
        return prob_map
    if fmt == "base64":
        return heatmap_to_base64(prob_map)
    if fmt == "png_bytes":
        return prob_map_to_png(prob_map)
    raise ValueError(f"Unsupported output_format: {output_format}")


def probability_heatmap(
    grid: Union[List[List[int]], np.ndarray],
    k: Optional[int],
    n_iter: int = 6000,
    *,
    seed: int = 0,
    sample_gamma: float = 0.0,
    history_dir: str = "samples",
    nearest_weight: float = 0.0,
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Heatmap simulation using :func:`simulate_full_board`.

    Parameters
    ----------
    grid : List[List[int]] or np.ndarray
        Board matrix with ``-1`` for unknown cells.
    k : int or None
        Target number to estimate.
    n_iter : int
        Simulation iterations.
    seed : int
        RNG seed for reproducibility.
    sample_gamma : float
        Weight for prior probabilities derived from ``history_dir``.
    history_dir : str
        Directory containing sample ``*.zip`` files.
    nearest_weight : float
        Blend ratio for :func:`nearest_value_affinity` heatmap.
    """

    rng = np.random.default_rng(seed)
    grid_np = np.asarray(grid, dtype=int)
    prob_map_dict = simulate_full_board(grid_np, k, n_iter=n_iter, rng=rng)

    mask_ratio = float(np.mean(grid_np == -1))
    gamma = sample_gamma * (1.0 + mask_ratio)

    if gamma > 0.0 and k is not None:
        try:
            pos_probs = compute_position_probabilities(history_dir, *grid_np.shape)
        except Exception as exc:  # pragma: no cover - history load failures
            logger.error("heatmap prior load failed: %s", exc)
            pos_probs = {}
    else:
        pos_probs = {}

    if k is not None:
        out = np.zeros_like(grid_np, dtype=float)
        for (r, c), cell in prob_map_dict.items():
            val = cell.get(k, 0.0)
            if pos_probs:
                val = (1.0 - gamma) * val + gamma * pos_probs.get((r, c), {}).get(
                    k, 0.0
                )
            out[r, c] = val
        if nearest_weight > 0:
            from modules import nearest_value_affinity

            near = nearest_value_affinity(grid_np, k, tolerance=1, radius=1)
            out = (1.0 - nearest_weight) * out + nearest_weight * near
        if out.max() > 0:
            out = out / float(out.max())
        return out

    numbers = {n for cell in prob_map_dict.values() for n in cell}
    result: Dict[int, np.ndarray] = {
        int(n): np.zeros_like(grid_np, dtype=float) for n in numbers
    }
    for (r, c), cell in prob_map_dict.items():
        for n, p in cell.items():
            result[int(n)][r, c] = p
    return result


def evaluate_prediction_accuracy(num_trials: int = 50, seed: int = 0) -> float:
    """Return top-1 accuracy over randomly generated boards."""

    rng = np.random.default_rng(seed)
    hits = 0
    trials = 0
    for _ in range(num_trials):
        rows = int(rng.integers(4, 21))
        cols = int(rng.integers(5, 21))
        full = generate_unique_grid(rows, cols, rng=rng)
        mask = rng.random((rows, cols)) < 0.5
        board = full.copy()
        board[mask] = -1
        blanks = np.argwhere(board == -1)
        if blanks.size == 0:
            continue
        r, c = blanks[rng.integers(len(blanks))]
        target = int(full[r, c])
        board[r, c] = -1
        res = predict_scratch_card(
            board.tolist(),
            target_num=target,
            iterations=20,
            global_iter=10,
            focus_iter=5,
            top_n=5,
            epsilon=0.1,
        )
        preds = res.get("predictions", [])
        trials += 1
        if preds and preds[0]["row"] == r and preds[0]["col"] == c:
            hits += 1
    if trials == 0:
        return 0.0
    return hits / float(trials)
