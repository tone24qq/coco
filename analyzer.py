import os
import math
import numpy as np
import xxhash
from scipy.stats import qmc
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple, Any, Optional
from joblib import Parallel, delayed
import logging
import heapq

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from modules import FORMULA_REGISTRY, compute_global_features
from brain import (
    EXT_GM20_Skip_Pattern_Confidence_Vec,
    MathUtils,
    BoardAnalyzerUtils,
    REGISTERED_MODULES_BRAIN,
    get_module_score,
    bytes_to_grid
)

math_utils = MathUtils()
analyzer_utils = BoardAnalyzerUtils()

# Count-Min Sketch (optimized for low memory)
class CountMinSketch:
    def __init__(self, width: int = 1024, depth: int = 1):
        self.w = max(1024, min(2048, int(8e9 / (depth * 4))))  # 8 GB RAM 動態調整
        self.d = depth
        self.table = np.zeros((depth, self.w), dtype=np.uint32)
        self.seeds = [i * 0x9e3779B1 for i in range(depth)]

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

def adjust_weights_based_on_history(history: Dict[str, float]) -> np.ndarray:
    """Dynamically adjust formula weights based on historical performance."""
    total = sum(history.values()) or 1e-10
    return np.array([history.get(f, 0.0) / total for f in ("random_entropy", "shuffle", "tail_cluster")])

def select_modules(grid: np.ndarray) -> List[str]:
    """Dynamically select modules based on grid characteristics."""
    base_modules = ["EXT_Q1_ProximityEntropy_Vec", "EXT_Q2_PotentialPath_Vec"]
    scores = {mod: np.mean(get_module_score(mod, grid)) for mod in REGISTERED_MODULES_BRAIN}
    top_modules = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:2]
    return base_modules + [m for m in top_modules if m not in base_modules]

@lru_cache(maxsize=500000)
def _cached_board(mask_key: str, seed: int, r: int, c: int,
                  kv_bytes: bytes, mask_bytes: bytes):
    """Return 1-D board (length r*c) with known cells filled."""
    sob = qmc.Sobol(d=r*c, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(r*c*4)))  # 計算 power-of-2 批次
    vec = sob.random_base2(m=m)[-1]   # 使用最後一點以消除警告
    flat = np.argsort(vec) + 1        # permutation 1..r*c

    known_mask = np.frombuffer(mask_bytes, dtype=bool)
    known_vals = np.frombuffer(kv_bytes, dtype=np.int32)

    flat[known_mask] = known_vals
    return flat                       # caller reshapes

def generate_full_boards(rows: int, cols: int, batch: int, rng: np.random.Generator, formulas: Tuple[str, ...], weights: np.ndarray) -> np.ndarray:
    """Generate batch of complete boards using weighted formulas with importance sampling."""
    n = rows * cols
    choices = rng.choice(formulas, size=batch, p=weights)
    boards = np.empty((batch, rows, cols), dtype=np.int16)
    known_vals = grid.ravel() if 'grid' in globals() else np.zeros(n, dtype=np.int16)
    known_mask = (grid != -1).ravel() if 'grid' in globals() else np.zeros(n, dtype=bool)
    kv_bytes = known_vals.tobytes()   # 轉換為 bytes
    mask_bytes = known_mask.tobytes() # 轉換為 bytes
    mask = xxhash.xxh64(kv_bytes + mask_bytes).hexdigest()
    seed = rng.integers(0, 0xFFFF)
    for i in range(batch):
        board1d = _cached_board(
            mask, seed & 0xFFFF,
            rows, cols,
            kv_bytes, mask_bytes      # no shape arg
        ).reshape(rows, cols)
        boards[i] = board1d
    return boards

def simulate_full_board(grid: np.ndarray, target_num: Optional[int], n_iter: int = 6000, rng: Optional[np.random.Generator] = None) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Simulate full boards with enhanced importance sampling and target_num hits."""
    if rng is None:
        rng = np.random.default_rng()

    g = np.asarray(grid, dtype=np.int16)
    rows, cols = g.shape
    blanks = np.argwhere(g == -1)
    known = np.argwhere(g != -1)
    known_vals = g[g != -1]
    legal_all = analyzer_utils.get_legal_values_for_placement(g)

    # Enhanced module selection for importance sampling
    modules = select_modules(g)
    module_scores = np.mean([get_module_score(mod, g) for mod in modules], axis=0)
    importance_weights = np.where(g == -1, module_scores, 0).flatten()
    importance_weights = importance_weights / (np.sum(importance_weights) + 1e-10)

    # Dynamic formula weights based on grid pattern
    history = {"random_entropy": 0.4, "shuffle": 0.3, "tail_cluster": 0.3}  # Default
    if np.mean(module_scores) > 0.6:  # Adjust if strong patterns detected
        history["tail_cluster"] += 0.1
        history["random_entropy"] -= 0.05
    weights = adjust_weights_based_on_history(history)

    formulas = ("random_entropy", "shuffle", "tail_cluster")
    remain = n_iter
    counts = defaultdict(lambda: defaultdict(int))

    while remain > 0:
        batch = min(4000, remain)
        boards = generate_full_boards(rows, cols, batch, rng, formulas, weights)

        if known.size:
            mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
            boards = boards[mask]
            if len(boards) == 0:
                batch = min(batch * 2, 8000)
                boards = generate_full_boards(rows, cols, batch, rng, formulas, weights)
                mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
                boards = boards[mask]

        if len(boards) > 0:
            for i, board in enumerate(boards):
                fast = 0.5 * get_module_score("EXT_Q1_ProximityEntropy_Vec", board) + 0.5 * get_module_score("EXT_Q2_PotentialPath_Vec", board)
                for r, c in blanks:
                    idx = r * cols + c
                    if rng.random() < importance_weights[idx]:
                        num = board[r, c]
                        counts[(r, c)][num] += fast[r, c]
                        if target_num is not None and num == target_num:
                            counts[(r, c)][target_num] += 2 * fast[r, c]
        remain -= batch

    prob_map = {}
    for (r, c) in [tuple(b) for b in blanks]:
        total = sum(counts[(r, c)].values()) or 1e-10
        probs = {n: max(counts[(r, c)][n] / total, 1e-10) for n in legal_all}
        prob_map[(r, c)] = probs

    # Two-phase scoring: Re-rank top K candidates
    candidates = [(r, c, max(probs.values()), num) for (r, c), probs in prob_map.items() for num in probs]
    top_k = heapq.nlargest(int(os.getenv("TOPK_RERANK", "100")), candidates, key=lambda x: x[2])
    final_prob_map = {}
    for r, c, fast_score, num in top_k:
        final_score = 0.5 * fast_score + 0.5 * (0.6 * get_module_score("EXT_Q3_DiscontinuitySym_Vec", g)[r, c] + 0.4 * get_module_score("EXT_Q4_ControlComposite_Vec", g)[r, c])
        if (r, c) not in final_prob_map:
            final_prob_map[(r, c)] = {}
        final_prob_map[(r, c)][num] = final_score

    return final_prob_map

def weight_prob_by_modules(grid: np.ndarray,
                           prob_map: Dict[Tuple[int, int], Dict[int, float]],
                           target_num: Optional[int] = None) -> Dict[Tuple[int, int], Dict[int, float]]:
    if not isinstance(prob_map, dict):
        logger.error(f"Invalid prob_map type: {type(prob_map)}")
        return {}

    result = prob_map.copy()
    modules = select_modules(grid)
    module_scores = Parallel(n_jobs=4)(
        delayed(get_module_score)(mod, grid) for mod in modules
    )
    module_scores = np.array(module_scores)

    for (r, c), probs in result.items():
        if (r, c) not in prob_map:
            continue
        scores = module_scores[:, r, c]
        scores = np.nan_to_num(scores, nan=0.0)
        softmax_scores = np.exp(scores / 0.5) / (np.sum(np.exp(scores / 0.5)) + 1e-10)
        mean_score = np.mean(softmax_scores)

        if target_num is not None:
            if target_num in probs:
                probs[target_num] = max(probs[target_num] * mean_score, 1e-10)
                total = probs[target_num] or 1e-10
                result[(r, c)] = {target_num: probs[target_num] / total}
            else:
                result[(r, c)] = {target_num: 0.0}
        else:
            for val in probs:
                probs[val] = max(probs[val] * mean_score, 1e-10)
            total = sum(probs.values()) or 1e-10
            result[(r, c)] = {k: v / total for k, v in probs.items()}

    return result

def global_unique(prob_map: Dict[Tuple[int, int], Dict[int, float]],
                  blanks: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        nums = sorted({n for d in prob_map.values() for n in d})
        cost = np.full((len(blanks), len(nums)), 50.0)

        for i, cell in enumerate(blanks):
            for j, n in enumerate(nums):
                prob = max(prob_map[cell].get(n, 1e-10), 1e-10)
                cost[i, j] = -math.log(prob)

        row, col = linear_sum_assignment(cost)
        return {blanks[r]: (nums[c], prob_map[blanks[r]].get(nums[c], 0.0))
                for r, c in zip(row, col)}
    except Exception as e:
        logger.error(f"Global unique assignment failed: {e}")
        assigned, res = set(), {}
        for cell in sorted(blanks,
                           key=lambda p: max(prob_map[p].values() or [0]),
                           reverse=True):
            for n, p in sorted(prob_map[cell].items(),
                               key=lambda x: x[1], reverse=True):
                if n not in assigned:
                    assigned.add(n)
                    res[cell] = (n, p)
                    break
            if cell not in res:
                res[cell] = (list(prob_map[cell].keys())[0], 0.0) if prob_map[cell] else (1, 0.0)
        return res

class MCTSNode:
    def __init__(self, grid, parent=None, parent_action=None):
        self.grid = grid.copy()
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.virtual_loss = 0
        self.untried_actions = [(r, c, v)
                                for r, c in np.argwhere(grid == -1)
                                for v in analyzer_utils.get_legal_values_for_placement(grid)]

    def uct_select(self, c_param=1.414):
        return max(self.children,
                   key=lambda c: (c.value / (c.visits + c.virtual_loss)) +
                   c_param * np.sqrt(2 * np.log(self.visits + 1e-10) / (c.visits + c.virtual_loss + 1e-10)) -
                   1.0 * c.virtual_loss)

def mcts(grid: np.ndarray, iterations: int = 1000):
    rows, cols = grid.shape
    root = MCTSNode(grid)

    def simulate(node):
        current = node
        while current.untried_actions and len(current.children) < 1.5 * current.visits ** 0.5:
            current = current.uct_select()
            current.virtual_loss += 1
        if current.untried_actions:
            r, c, v = current.untried_actions.pop()
            new_grid = current.grid.copy()
            new_grid[r, c] = v
            child = MCTSNode(new_grid, current, (r, c, v))
            current.children.append(child)
            current = child

        sim_result = simulate_full_board(current.grid, None, n_iter=100)
        if not isinstance(sim_result, dict):
            logger.error(f"Invalid sim_result type: {type(sim_result)}")
            return 0.0

        reward = 0.0
        for r, c in np.argwhere(grid == -1):
            if (r, c) in sim_result:
                weighted = weight_prob_by_modules(
                    current.grid, {(r, c): sim_result[(r, c)]})
                reward += max(weighted[(r, c)].values())

        while current is not None:
            current.visits += 1
            current.value += reward
            current.virtual_loss -= 1
            current = current.parent
        return reward

    Parallel(n_jobs=4)(delayed(simulate)(root) for _ in range(iterations // 4))
    best_child = max(root.children, key=lambda c: c.value / c.visits,
                     default=root)
    return best_child.grid

# Main prediction entry point
def predict_scratch_card(
    grid: List[List[int]],
    target_num: Optional[int] = None,
    iterations: Optional[int] = None,
    quick_iter: Optional[int] = None,
    refine_iter: Optional[int] = None,
    min_total_iter: Optional[int] = None,
    unique: bool = True
) -> Dict[str, Any]:
    grid_np = np.array(grid, dtype=np.int64)
    rows, cols = grid_np.shape
    blanks = [tuple(b) for b in np.argwhere(grid_np == -1)]

    if not blanks:
        return {"mode": "no_blanks", "predictions": [], "full_probabilities": {}}

    modules = [
        ("EXT_Q1_ProximityEntropy_Vec", "Proximity and entropy scoring"),
        ("EXT_Q2_PotentialPath_Vec", "Sequence and path scoring"),
        ("EXT_Q3_DiscontinuitySym_Vec", "Discontinuity and symmetry scoring"),
        ("EXT_Q4_ControlComposite_Vec", "Control and error correction")
    ]

    base_iter = iterations if iterations is not None else int(os.getenv("ITER", 6000))
    total_iter = int(base_iter * max(rows * cols / 40, 1))
    quick_iter = quick_iter if quick_iter is not None else int(total_iter * 0.35)
    refine_iter = refine_iter if refine_iter is not None else total_iter - quick_iter
    min_total_iter = min_total_iter if min_total_iter is not None else max(1000, total_iter // 5)

    logger.info(f"Simulating full board with {total_iter} iterations")
    prob_map = simulate_full_board(
        grid_np, target_num, n_iter=total_iter, rng=np.random.default_rng()
    )

    if target_num is not None:
        rank = [{
            "row": r,
            "col": c,
            "candidates": [target_num],
            "probability": prob_map[(r, c)].get(target_num, 0.0) * 100
        } for r, c in blanks]
        rank.sort(key=lambda x: x["probability"], reverse=True)

        module_scores = {mod: get_module_score(mod, grid_np) for mod, _ in modules}
        for pred in rank[:3]:
            reasons = []
            scores = [(mod, module_scores[mod][pred['row'], pred['col']], desc)
                      for mod, desc in modules]
            top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for mod, score, desc in top_modules:
                if score > 0.5:
                    reasons.append(f"{desc} (score: {score:.2f})")
            pred["reasons"] = reasons if reasons else ["No dominant module contribution"]

        return {
            "mode": "target",
            "target": target_num,
            "predictions": rank[:3],
            "full_probabilities": prob_map
        }

    if unique:
        assign = global_unique(prob_map, blanks)
        best_grid = mcts(grid_np, iterations=1000)

        old_conf = max([p for (_, _), (_, p) in assign.items()] or [0])
        new_conf = max([
            max(weight_prob_by_modules(best_grid, {(r, c): prob_map[(r, c)]})[(r, c)].values())
            for r, c in blanks
        ] or [0])

        if new_conf <= old_conf * 0.95:
            preds = [{
                "row": r,
                "col": c,
                "candidates": [n],
                "probability": float(p) * 100
            } for (r, c), (n, p) in assign.items()]
            mode = "unique"
        else:
            preds = process_grid(best_grid)
            mode = "mcts_unique"

        module_scores = {mod: get_module_score(mod, grid_np) for mod, _ in modules}
        for pred in preds[:3]:
            reasons = []
            scores = [(mod, module_scores[mod][pred['row'], pred['col']], desc)
                      for mod, desc in modules]
            top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for mod, score, desc in top_modules:
                if score > 0.5:
                    reasons.append(f"{desc} (score: {score:.2f})")
            pred["reasons"] = reasons if reasons else ["No dominant module contribution"]

        preds.sort(key=lambda x: x["probability"], reverse=True)
        return {
            "mode": mode,
            "predictions": preds[:3],
            "full_probabilities": prob_map
        }

    preds = []
    for (r, c), dist in prob_map.items():
        top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        nums, probs = zip(*top3) if top3 else ([], [])
        preds.append({
            "row": r,
            "col": c,
            "candidates": list(nums),
            "probability": [p * 100 for p in probs]
        })

    module_scores = {mod: get_module_score(mod, grid_np) for mod, _ in modules}
    for pred in preds[:3]:
        reasons = []
        scores = [(mod, module_scores[mod][pred['row'], pred['col']], desc)
                  for mod, desc in modules]
        top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        for mod, score, desc in top_modules:
            if score > 0.5:
                reasons.append(f"{desc} (score: {score:.2f})")
        pred["reasons"] = reasons if reasons else ["No dominant module contribution"]

    preds.sort(key=lambda x: x["probability"][0] if x["probability"] else 0,
               reverse=True)
    return {
        "mode": "top3",
        "predictions": preds[:3],
        "full_probabilities": prob_map
    }

def process_grid(grid):
    blanks = np.argwhere(grid == -1)
    preds = []
    for r, c in blanks:
        legal_vals = analyzer_utils.get_legal_values_for_placement(grid)
        max_prob = max(legal_vals) if legal_vals else 1
        preds.append({
            "row": int(r),
            "col": int(c),
            "candidates": [int(max_prob)],
            "probability": 100.0 if grid[r, c] != -1 else 50.0
        })
    return preds