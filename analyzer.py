import base64
import heapq
import json
import logging
import math
import os
import sys
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import xxhash
from joblib import Parallel, delayed

import brain
# fmt: off
from brain import (AGG_WEIGHTS, REGISTERED_MODULES_BRAIN, BoardAnalyzerUtils,
                   EXT_GM20_Skip_Pattern_Confidence_Vec, MathUtils,
                   aggregate_scores, bytes_to_grid, get_module_score)
# fmt: on
from modules import FORMULA_REGISTRY

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

math_utils = MathUtils()
analyzer_utils = BoardAnalyzerUtils()


# 來自 probmap_key_patch_v2.txt
def _native_coord(k):
    return int(k[0]), int(k[1])


def _native_dict(d):
    return {_native_coord(k): v for k, v in d.items()}


def _iter_json_from_zip(zip_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a zip file with basic validation."""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as f:
                    data = json.load(f)
            except Exception as exc:  # pragma: no cover - corrupted JSON
                logger.error("Failed to read %s:%s: %s", zip_path.name, name, exc)
                continue
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
            yield {"rows": rows, "cols": cols, "grid": grid}
    logger.info("Loaded %s (%d JSON)", zip_path.name, count)


def iter_sample_jsons(samples_dir: str) -> Iterator[Dict[str, Any]]:
    """Iterate through all JSON samples in ``samples_dir``."""
    path = Path(samples_dir)
    zip_count = 0
    json_count = 0
    for zp in sorted(path.glob("*.zip")):
        zip_count += 1
        try:
            for item in _iter_json_from_zip(zp):
                json_count += 1
                yield item
        except Exception as exc:  # pragma: no cover - broken zip
            logger.error("Failed to load %s: %s", zp.name, exc)
    logger.info("Total loaded: %d zip files, %d JSON", zip_count, json_count)


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


@lru_cache(maxsize=8)
def compute_position_probabilities(
    samples_dir: str, rows: int, cols: int
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Return per-cell number probabilities from all samples."""
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


# Count-Min Sketch (optimized for low memory)
class CountMinSketch:
    def __init__(self, width: int = 1024, depth: int = 1):
        self.w = max(1024, min(2048, int(8e9 / (depth * 4))))  # 8 GB RAM 動態調整
        self.d = depth
        self.table = np.zeros((depth, self.w), dtype=np.uint32)
        self.seeds = [i * 0x9E3779B1 for i in range(depth)]

    def _idx(self, key: bytes, seed: int) -> int:
        return xxhash.xxh32(key, seed=seed).intdigest() % self.w

    def update(self, key: bytes, value: int = 1):
        for i, s in enumerate(self.seeds):
            self.table[i, self._idx(key, s)] += value

    def query(self, key: bytes) -> int:
        return min(self.table[i, self._idx(key, s)] for i, s in enumerate(self.seeds))


def pack_key(cell_idx: int, num: int) -> bytes:
    return (cell_idx << 16 | num).to_bytes(4, "little")


# Precompute skip scores with LRU cache
@lru_cache(maxsize=1024)
def precompute_skip_scores(grid_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    grid = bytes_to_grid(grid_bytes, (rows, cols))
    return EXT_GM20_Skip_Pattern_Confidence_Vec(grid)


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
        mods = list(REGISTERED_MODULES_BRAIN)
    else:
        base_modules = brain.get_core_modules()
        scores = {
            m: np.mean(get_module_score(m, grid, target=target)) for m in base_modules
        }
        mods = sorted(scores, key=scores.get, reverse=True)

    if "EXT_Q12_ArithmeticProgression_Vec" not in mods:
        mods.append("EXT_Q12_ArithmeticProgression_Vec")
    if target is not None and "EXT_Q11_GlobalDigitAffinity_Vec" not in mods:
        mods.append("EXT_Q11_GlobalDigitAffinity_Vec")
    if target is not None and "EXT_Q14_TargetAffinity_Vec" not in mods:
        mods.append("EXT_Q14_TargetAffinity_Vec")
    if "EXT_M12_RestoreOriginalValue_Vec" not in mods:
        mods.append("EXT_M12_RestoreOriginalValue_Vec")
    if "EXT_Q15_GlobalSpread_Vec" not in mods:
        mods.append("EXT_Q15_GlobalSpread_Vec")
    if "EXT_Q16_NumericalRelationalPattern_Vec" not in mods:
        mods.append("EXT_Q16_NumericalRelationalPattern_Vec")
    if (
        os.getenv("ENABLE_SPECTRUM", "0") == "1"
        and "EXT_Q13_GlobalConsistencySpectrum_Vec" not in mods
    ):
        mods.append("EXT_Q13_GlobalConsistencySpectrum_Vec")
    if "EXT_M11_Mirror_Sequence_Vec" not in mods:
        mods.append("EXT_M11_Mirror_Sequence_Vec")
    return mods


@lru_cache(maxsize=500000)
def _cached_board(
    mask_key: str, seed: int, r: int, c: int, kv_bytes: bytes, idx_bytes: bytes
):
    """Return a unique 1-D board (length r*c) with known cells filled."""
    rng = np.random.default_rng(seed)
    n = r * c
    perm = rng.permutation(n) + 1

    idx = np.frombuffer(idx_bytes, dtype=np.int32)
    if idx.size == 0:
        return perm.astype(np.int16)

    vals = np.frombuffer(kv_bytes, dtype=np.int32)
    if idx.size != vals.size:
        return perm.astype(np.int16)

    # Remove known values from permutation to avoid duplicates
    mask = np.isin(perm, vals, invert=True)
    remaining = perm[mask]

    board = np.empty(n, dtype=np.int16)
    board[idx] = vals
    unknown_idx = np.setdiff1d(np.arange(n), idx, assume_unique=True)
    board[unknown_idx] = remaining[: unknown_idx.size]
    return board


def fill_unknowns_randomly(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fill -1 cells in ``grid`` with a random permutation of remaining numbers."""
    g = np.asarray(grid, dtype=int).copy()
    blanks = np.argwhere(g == -1)
    if blanks.size == 0:
        return g

    rows, cols = g.shape
    all_vals = np.arange(1, rows * cols + 1)
    remain = np.setdiff1d(all_vals, g[g != -1], assume_unique=True)
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
    known_vals = grid.ravel()
    known_mask = (grid != -1).ravel()
    kv_bytes = known_vals.tobytes()
    idx_bytes = known_mask.nonzero()[0].astype(np.int32).tobytes()
    mask = xxhash.xxh64(kv_bytes + idx_bytes).hexdigest()
    seed = rng.integers(0, 0xFFFF)
    for i in range(batch):
        board1d = _cached_board(
            mask, seed & 0xFFFF, rows, cols, kv_bytes, idx_bytes
        ).reshape(rows, cols)
        boards[i] = board1d
    return boards


def simulate_full_board(
    grid: np.ndarray,
    target_num: Optional[int],
    n_iter: int = 6000,
    rng: Optional[np.random.Generator] = None,
    *,
    focus_cells: Optional[List[Tuple[int, int]]] = None,
    epsilon: float = 0.0,
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
    blanks = np.argwhere(g == -1)
    known = np.argwhere(g != -1)
    known_vals = g[g != -1]
    legal_all = analyzer_utils.get_legal_values_for_placement(g)

    if target_num is not None:
        count_map = np.zeros((rows, cols), dtype=int)
        for _ in range(max(1, n_iter)):
            filled = fill_unknowns_randomly(g, rng)
            mask = filled == target_num
            count_map += mask.astype(int)

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
    importance_weights = np.where(g == -1, module_scores, 0).flatten()
    importance_weights = importance_weights / (np.sum(importance_weights) + 1e-10)

    # Dynamic formula weights based on grid pattern
    history = {"random_entropy": 0.4, "shuffle": 0.3, "tail_cluster": 0.3}
    if np.mean(module_scores) > 0.6:
        history["tail_cluster"] += 0.1
        history["random_entropy"] -= 0.05

    formulas = ("random_entropy", "shuffle", "tail_cluster")
    weights = adjust_weights_based_on_history(history, formulas)
    remain = n_iter
    counts = defaultdict(lambda: defaultdict(int))
    focus_set = {tuple(fc) for fc in focus_cells} if focus_cells else None
    other_cells = (
        [tuple(b) for b in blanks if tuple(b) not in focus_set] if focus_set else []
    )

    while remain > 0:
        batch = min(4000, remain)
        boards = generate_full_boards(rows, cols, batch, rng, formulas, weights, g)

        if known.size:
            mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
            boards = boards[mask]
            if len(boards) == 0:
                batch = min(batch * 2, 8000)
                boards = generate_full_boards(
                    rows, cols, batch, rng, formulas, weights, g
                )
                mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
                boards = boards[mask]

        if len(boards) > 0:
            for board in boards:
                mods_fast = [
                    "EXT_Q1_ProximityEntropy_Vec",
                    "EXT_Q2_PotentialPath_Vec",
                    "EXT_Q5_GlobalEntropy_Vec",
                    "EXT_Q6_LineBridge_Vec",
                    "EXT_Q7_VariancePrior_Vec",
                ]
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
                for r, c in cells_iter:
                    idx = r * cols + c
                    if rng.random() < importance_weights[idx]:
                        num = board[r, c]
                        counts[(r, c)][num] += fast[r, c]
                        if target_num is not None and num == target_num:
                            counts[(r, c)][num] += 2 * fast[r, c]
        remain -= batch

    prob_map = {}
    for r, c in [tuple(b) for b in blanks]:
        total = sum(counts[(r, c)].values()) or 1e-10
        probs = {n: max(counts[(r, c)][n] / total, 1e-10) for n in legal_all}
        prob_map[(r, c)] = probs

    # Two-phase scoring: Re-rank top K candidates with Borda or Soft-Max
    tau = float(os.getenv("TAU_SOFTMAX", "0.3"))
    mods_rerank = [
        "EXT_Q1_ProximityEntropy_Vec",
        "EXT_Q2_PotentialPath_Vec",
        "EXT_Q3_DiscontinuitySym_Vec",
        "EXT_Q4_ControlComposite_Vec",
        "EXT_Q5_GlobalEntropy_Vec",
        "EXT_Q6_LineBridge_Vec",
        "EXT_Q7_VariancePrior_Vec",
    ]
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
    final_prob_map = {}
    for r, c, fast_score, num in top_k:
        final_score = float(soft_heat[r, c])
        if (r, c) not in final_prob_map:
            final_prob_map[(r, c)] = {}
        final_prob_map[(r, c)][num] = final_score

    # 來自 probmap_key_patch_v2.txt
    prob_map = {(int(r), int(c)): cell for (r, c), cell in final_prob_map.items()}

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
) -> Dict[str, Any]:
    grid_np = np.array(grid, dtype=np.int64)
    rows, cols = grid_np.shape
    blanks = [
        tuple(map(int, b)) for b in np.argwhere(grid_np == -1)
    ]  # 來自 probmap_key_patch_v2.txt

    if not blanks:
        return {
            "mode": "no_blanks",
            "predictions": [],
            "full_probabilities": {},
        }

    modules = [
        ("EXT_Q1_ProximityEntropy_Vec", "Proximity and entropy scoring"),
        ("EXT_Q3_DiscontinuitySym_Vec", "Discontinuity and symmetry scoring"),
        ("EXT_Q5_GlobalEntropy_Vec", "Global entropy and clustering"),
        ("EXT_Q14_TargetAffinity_Vec", "Target number affinity"),
        ("EXT_Q15_GlobalSpread_Vec", "Global spread preference"),
        (
            "EXT_Q16_NumericalRelationalPattern_Vec",
            "Numerical relational patterns",
        ),
    ]

    mod_names = [m for m, _ in modules]
    weights = np.array([0.2, 0.15, 0.25, 0.2, 0.1, 0.2], dtype=float)
    score_stack = np.stack(
        [
            get_module_score(m, grid_np, target=target_num, priors=priors)
            for m in mod_names
        ],
        axis=0,
    )
    final_score_map = (weights[:, None, None] * score_stack).sum(axis=0)
    if gamma_history > 0.0 and target_num is not None:
        try:
            hist = compute_history_frequency(history_dir, target_num, rows, cols)
            if hist.max() > 0:
                final_score_map += gamma_history * (hist / float(hist.max()))
        except Exception as exc:  # pragma: no cover - history load failures
            logger.error("history frequency failed: %s", exc)

    if sample_gamma > 0.0:
        try:
            pos_probs = compute_position_probabilities(history_dir, rows, cols)
            sample_map = np.zeros((rows, cols), dtype=float)
            for (r, c), dist in pos_probs.items():
                if dist:
                    sample_map[r, c] = max(dist.values())
            if sample_map.max() > 0:
                sample_map /= float(sample_map.max())
                final_score_map += sample_gamma * sample_map
        except Exception as exc:  # pragma: no cover - history load failures
            logger.error("position frequency failed: %s", exc)

    phase1 = global_iter if global_iter is not None else iterations or 5000
    phase2 = focus_iter if focus_iter is not None else 1000
    top_k = result_top_k or int(os.getenv("RESULT_TOP_K", "3"))

    def _trim(items: List[Any]) -> List[Any]:
        if top_k is None or top_k <= 0 or top_k >= len(items):
            return items
        return items[:top_k]

    logger.info(
        "Two-phase | phase1=%d, phase2=%d, top_k=%d, top_n=%d, eps=%.6f",
        phase1,
        phase2,
        top_k,
        top_n,
        epsilon,
    )

    prob_map = simulate_full_board(
        grid_np,
        target_num,
        n_iter=phase1,
        rng=np.random.default_rng(),
    )

    # select top-n cells for refinement using weighted module scores
    ranked = sorted(
        [(r, c, final_score_map[r, c]) for r, c in blanks],
        key=lambda x: x[2],
        reverse=True,
    )
    focus_cells = [(r, c) for r, c, _ in ranked[: max(1, min(top_n, len(ranked)))]]

    if phase2 > 0:
        refine_map = simulate_full_board(
            grid_np,
            target_num,
            n_iter=phase2,
            rng=np.random.default_rng(),
            focus_cells=focus_cells,
            epsilon=epsilon,
        )
        prob_map.update(refine_map)

    if sample_gamma > 0.0:
        try:
            pos_probs = compute_position_probabilities(history_dir, rows, cols)
            for key, dist in prob_map.items():
                prior = pos_probs.get(key, {})
                for num in list(dist.keys()):
                    dist[num] *= 1.0 + sample_gamma * prior.get(num, 0.0)
                tot = sum(dist.values()) or 1e-10
                for num in dist:
                    dist[num] /= tot
        except Exception as exc:  # pragma: no cover - history load failures
            logger.error("position frequency blend failed: %s", exc)

    # 后置正規化避免混權後機率失真
    for dist in prob_map.values():
        total = sum(dist.values()) or 1e-12
        for k in dist:
            dist[k] /= total

    module_scores = {
        mod: get_module_score(mod, grid_np, priors=priors, target=target_num)
        for mod, _ in modules
    }
    logger.info(
        "module_scores: %s",
        {m: float(np.mean(v)) for m, v in module_scores.items()},
    )
    top3 = sorted(
        ((k, max(v.values())) for k, v in prob_map.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:3]
    summary = ", ".join(f"{r + 1}行{c + 1}列={p:.2f}" for (r, c), p in top3)
    logger.info("prob_map top3: %s", summary)

    if target_num is not None:
        for key, p in prob_map.items():
            prob_map[key] = {target_num: p.get(target_num, 0.0)}

    if target_num is not None:
        rank = [
            {
                "row": r,
                "col": c,
                "candidates": [target_num],
                "probability": prob_map.get((r, c), {}).get(target_num, 0.0) * 100,
            }
            for r, c in prob_map.keys()
        ]  # 改用 .get()
        rank.sort(key=lambda x: x["probability"], reverse=True)

        module_scores = {
            mod: get_module_score(mod, grid_np, priors=priors, target=target_num)
            for mod, _ in modules
        }
        for pred in rank[:3]:
            reasons = []
            scores = [
                (mod, module_scores[mod][pred["row"], pred["col"]], desc)
                for mod, desc in modules
            ]
            top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for mod, score, desc in top_modules:
                if score > 0.5:
                    reasons.append(f"{desc} (score: {score:.2f})")
            pred["reasons"] = (
                reasons if reasons else ["No dominant module contribution"]
            )
            pred["module_scores"] = {
                mod: float(module_scores[mod][pred["row"], pred["col"]])
                for mod, _ in modules
            }

        return {
            "mode": "target",
            "target": target_num,
            "predictions": _trim(rank),
            "full_probabilities": prob_map,
        }

    if unique and target_num is None:
        assign = global_unique(prob_map, blanks)
        best_grid = mcts(grid_np, iterations=1000)

        # -------------------------------
        # 重新計算信心度 (old_conf / new_conf)
        # -------------------------------
        # prob_map : Dict[(row, col), Dict[num, prob]]
        # best_grid: Numpy 2-D array or similar
        # candidates: List[(row, col)]
        # 1️⃣ 先找整張表目前“最高”的機率 (baseline)
        old_conf = max(
            p
            for _, cell_probs in prob_map.items()
            for p in cell_probs.values()  # 逐格取出內部 dict  # 逐個號碼機率
        )

        # 2️⃣ 嘗試把每個候選格 (r,c) 掛回去後，重新加權 → 看能否誕生更高機率
        new_conf = max(
            [
                max(
                    weight_prob_by_modules(  # <<< 你自己已有的函式
                        best_grid,  # - 當前最佳棋盤
                        {(r, c): prob_map.get((r, c), {})},  # - 只帶單一候選格的機率
                    )
                    .get((r, c), {})
                    .values(),  # 取回各模組加權後的機率們
                    default=0,  # ← dict 為空時避免 ValueError
                )
                for (r, c) in blanks  # 逐一測試所有候選格
            ]
        )

        if new_conf <= old_conf * 0.95:
            preds = [
                {
                    "row": r,
                    "col": c,
                    "candidates": [n],
                    "probability": float(p) * 100,
                }
                for (r, c), (n, p) in assign.items()
            ]
            mode = "unique"
        else:
            preds = process_grid(best_grid)
            mode = "mcts_unique"

        module_scores = {
            mod: get_module_score(mod, grid_np, priors=priors, target=target_num)
            for mod, _ in modules
        }
        for pred in preds[:3]:
            reasons = []
            scores = [
                (mod, module_scores[mod][pred["row"], pred["col"]], desc)
                for mod, desc in modules
            ]
            top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for mod, score, desc in top_modules:
                if score > 0.5:
                    reasons.append(f"{desc} (score: {score:.2f})")
            pred["reasons"] = (
                reasons if reasons else ["No dominant module contribution"]
            )
            pred["module_scores"] = {
                mod: float(module_scores[mod][pred["row"], pred["col"]])
                for mod, _ in modules
            }

        preds.sort(key=lambda x: x["probability"], reverse=True)
        return {
            "mode": mode,
            "predictions": _trim(preds),
            "full_probabilities": prob_map,
        }

    preds = []
    for (r, c), dist in prob_map.items():
        top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, probs = zip(*top3) if top3 else ([], [])
        preds.append(
            {
                "row": r,
                "col": c,
                "candidates": list(nums),
                "probability": [p * 100 for p in probs],
            }
        )

    module_scores = {
        mod: get_module_score(mod, grid_np, priors=priors, target=target_num)
        for mod, _ in modules
    }
    for pred in preds[:3]:
        reasons = []
        scores = [
            (mod, module_scores[mod][pred["row"], pred["col"]], desc)
            for mod, desc in modules
        ]
        top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        for mod, score, desc in top_modules:
            if score > 0.5:
                reasons.append(f"{desc} (score: {score:.2f})")
        pred["reasons"] = reasons if reasons else ["No dominant module contribution"]
        pred["module_scores"] = {
            mod: float(module_scores[mod][pred["row"], pred["col"]])
            for mod, _ in modules
        }

    preds.sort(
        key=lambda x: x["probability"][0] if x["probability"] else 0,
        reverse=True,
    )
    return {
        "mode": "top3",
        "predictions": _trim(preds),
        "full_probabilities": prob_map,
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
    else:
        counts = {int(val): np.zeros((rows, cols), dtype=int) for val in remain}

    for _ in range(max(1, n_iter)):
        sample = rng.permutation(remain)
        board = g.copy()
        board[blank_idx] = sample[: blanks.shape[0]]

        if k is not None:
            hits = board == k
            counts += hits.astype(int)
        else:
            for val in remain:
                counts[int(val)] += (board == val).astype(int)

    if k is not None:
        prob = counts.astype(float) / float(n_iter)
        prob[g != -1] = 0.0
        return prob

    prob_all: Dict[int, np.ndarray] = {}
    for val, mat in counts.items():
        arr = mat.astype(float) / float(n_iter)
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
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Heatmap simulation using :func:`simulate_full_board`."""

    rng = np.random.default_rng(seed)
    grid_np = np.asarray(grid, dtype=int)
    prob_map_dict = simulate_full_board(grid_np, k, n_iter=n_iter, rng=rng)

    if k is not None:
        out = np.zeros_like(grid_np, dtype=float)
        for (r, c), cell in prob_map_dict.items():
            out[r, c] = cell.get(k, 0.0)
        return out

    numbers = {n for cell in prob_map_dict.values() for n in cell}
    result: Dict[int, np.ndarray] = {
        int(n): np.zeros_like(grid_np, dtype=float) for n in numbers
    }
    for (r, c), cell in prob_map_dict.items():
        for n, p in cell.items():
            result[int(n)][r, c] = p
    return result
