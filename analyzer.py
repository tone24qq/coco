import os
import math
import numpy as np
import xxhash
from scipy.stats import qmc
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple, Any, Optional
from joblib import Parallel, delayed
import logging
import ray
from numba import njit
import multiprocessing
from multiprocessing import shared_memory
import psutil

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
    get_module_score
)

math_utils = MathUtils()
analyzer_utils = BoardAnalyzerUtils()

# Count-Min Sketch (optimized for low memory)
class CountMinSketch:
    def __init__(self, width: int = 1024, depth: int = 1):
        self.w = max(1024, min(2048, int(8e9 / (depth * 4))))
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
@lru_cache(maxsize=300)
def precompute_skip_scores(grid_bytes: bytes, rows: int, cols: int) -> np.ndarray:
    grid = np.frombuffer(grid_bytes, dtype=np.int16).reshape(rows, cols)
    return EXT_GM20_Skip_Pattern_Confidence_Vec(grid)

def adjust_weights_based_on_history(history: Dict[str, float]) -> np.ndarray:
    """Dynamically adjust formula weights based on historical performance."""
    total = sum(history.values()) or 1e-10
    weights = np.array([history.get(f, 0.0) / total for f in ("random_entropy", "shuffle", "tail_cluster")], dtype=np.float32)
    weights = weights / (np.sum(weights) + 1e-10)  # Normalize weights
    return weights

def select_modules(grid: np.ndarray) -> List[str]:
    """Dynamically select modules based on grid characteristics."""
    base_modules = ["EXT_M1_Tail_Pattern_Vec", "EXT_GM20_Skip_Pattern_Confidence_Vec"]
    scores = {mod: np.mean(get_module_score(mod, grid)) for mod in REGISTERED_MODULES_BRAIN}
    top_modules = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:2]
    return base_modules + [m for m in top_modules if m not in base_modules]

def reverse_engineer_seed(grid: np.ndarray, known: np.ndarray, known_vals: np.ndarray, rows: int, cols: int, seed_range: Tuple[int, int] = (0, 100000)) -> Tuple[int, float]:
    """Reverse engineer the seed for excel formula using coarse-to-fine search."""
    def loss_function(seed: int, grid: np.ndarray, known: np.ndarray, known_vals: np.ndarray) -> float:
        rng = np.random.default_rng(seed)
        generated = FORMULA_REGISTRY["excel"](rows, cols, rng)
        mse = np.mean((generated[known[:, 0], known[:, 1]] - known_vals) ** 2)
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in np.unique(generated, return_counts=True)[1] / (rows * cols)])
        return mse + abs(entropy - compute_global_features(grid)[1])

    coarse_seeds = np.arange(max(0, seed_range[0]), seed_range[1], 500, dtype=np.int32)
    coarse_losses = Parallel(n_jobs=4)(delayed(loss_function)(s, grid, known, known_vals) for s in coarse_seeds)
    top_k = np.argsort(coarse_losses)[:3]  # Limit to top-3 seeds
    best_coarse_seed = coarse_seeds[top_k[0]]
    best_coarse_loss = coarse_losses[top_k[0]]

    def fine_loss(seed) -> float:
        seed_scalar = float(seed[0]) if isinstance(seed, np.ndarray) else float(seed)
        seed_int = max(0, int(round(seed_scalar)))
        return loss_function(seed_int, grid, known, known_vals)

    result = minimize(
        fine_loss,
        x0=[best_coarse_seed],
        method='Powell',
        bounds=[(max(0, best_coarse_seed - 2500), best_coarse_seed + 2500)],
    )

    best_seed = max(0, int(round(result.x[0])))
    best_loss = result.fun
    logger.info(f"Reverse engineered seed: {best_seed}, loss: {best_loss:.4f}")
    return best_seed, best_loss

class SurrogateModel:
    """Gaussian Process surrogate model for seed prediction."""
    def __init__(self):
        self.gp = GaussianProcessRegressor(kernel=RBF(length_scale=1000), n_restarts_optimizer=5)

    def fit(self, seeds: np.ndarray, losses: np.ndarray):
        """Fit the surrogate model with seed-loss pairs."""
        self.gp.fit(seeds.reshape(-1, 1), losses)

    def predict(self, seeds: np.ndarray) -> np.ndarray:
        """Predict loss for new seeds."""
        return self.gp.predict(seeds.reshape(-1, 1))

    def select_top_seeds(self, seed_range: Tuple[int, int], n_samples: int = 100) -> List[int]:
        """Select top seeds using active learning."""
        seeds = np.linspace(seed_range[0], seed_range[1], n_samples, dtype=np.int32)
        pred_losses, std = self.gp.predict(seeds.reshape(-1, 1), return_std=True)
        top_indices = np.argsort(pred_losses + std)[:10]
        return seeds[top_indices].tolist()

def generate_full_boards(rows: int, cols: int, batch: int, rng: np.random.Generator, formulas: Tuple[str, ...], weights: np.ndarray) -> np.ndarray:
    """Generate batch of complete boards using weighted formulas with importance sampling."""
    n = rows * cols
    choices = rng.choice(formulas, size=batch, p=weights)
    boards = np.empty((batch, n), dtype=np.int16)
    for i, f in enumerate(choices):
        func = FORMULA_REGISTRY.get(f, FORMULA_REGISTRY[list(FORMULA_REGISTRY.keys())[0]])
        boards[i] = func(rows, cols, rng).ravel()
    return boards.reshape(batch, rows, cols)

@ray.remote
def simulate_full_board(grid: np.ndarray, target_num: Optional[int], n_iter: int = 6000, seed: Optional[int] = None) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Simulate full boards with enhanced importance sampling and target_num hits."""
    rng = np.random.default_rng(seed if seed is not None else np.random.randint(0, 1000000))
    
    shm_name = f"grid_{id(grid)}_{rng.integers(0, 1000000)}"
    shm = shared_memory.SharedMemory(create=True, size=grid.nbytes, name=shm_name)
    shared_grid = np.ndarray(grid.shape, dtype=np.int16, buffer=shm.buf)
    np.copyto(shared_grid, grid)
    
    try:
        g = shared_grid
        rows, cols = g.shape
        blanks = np.argwhere(g == -1)
        known = np.argwhere(g != -1)
        known_vals = g[g != -1]
        legal_all = analyzer_utils.get_legal_values_for_placement(g)

        modules = select_modules(g)
        module_scores = np.mean([get_module_score(mod, g) for mod in modules], axis=0)
        importance_weights = np.where(g == -1, module_scores, 0).flatten()
        importance_weights = importance_weights / (np.sum(importance_weights) + 1e-10)

        history = {"random_entropy": 0.4, "shuffle": 0.3, "tail_cluster": 0.3}
        if np.mean(module_scores) > 0.6:
            history["tail_cluster"] += 0.1
            history["random_entropy"] -= 0.05
        weights = adjust_weights_based_on_history(history)

        variance = np.var(module_scores[module_scores > 0]) if np.any(module_scores > 0) else 1.0
        adjusted_iter = int(n_iter * min(1.5, max(0.5, variance / 0.1)))

        preferred = ("random_entropy", "shuffle", "tail_cluster")
        formulas = tuple(f for f in preferred if f in FORMULA_REGISTRY) or tuple(FORMULA_REGISTRY.keys())

        formulas_and_weights = [(f, w) for f, w in zip(formulas, weights) if f in FORMULA_REGISTRY]
        if not formulas_and_weights:
            raise ValueError("No valid formulas found in FORMULA_REGISTRY.")
        formulas, weights = zip(*formulas_and_weights)
        weights = np.array(weights, dtype=np.float32) / (np.sum(weights) + 1e-10)  # Normalize weights

        remain = adjusted_iter
        counts = defaultdict(lambda: defaultdict(int))

        while remain > 0:
            batch = min(1000, remain)
            if psutil.virtual_memory().percent > 75:
                batch = max(100, batch // 2)
            boards = generate_full_boards(rows, cols, batch, rng, formulas, weights)

            if known.size:
                mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
                boards = boards[mask]
                if len(boards) == 0:
                    batch = min(batch * 2, 2000)
                    boards = generate_full_boards(rows, cols, batch, rng, formulas, weights)
                    mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
                    boards = boards[mask]

            if len(boards) > 0:
                for i, board in enumerate(boards):
                    for r, c in blanks:
                        idx = r * cols + c
                        if rng.random() < importance_weights[idx]:
                            num = board[r, c]
                            counts[(r, c)][num] += 1
                            if target_num is not None and num == target_num:
                                counts[(r, c)][num] += 2
            remain -= batch

        prob_map = {}
        for (r, c) in [tuple(b) for b in blanks]:
            total = sum(counts[(r, c)].values()) or 1e-10
            probs = {n: max(counts[(r, c)][n] / total, 1e-10) for n in legal_all}
            prob_map[(r, c)] = probs

        return prob_map
    finally:
        shm.close()
        shm.unlink()

@ray.remote
def simulate_mcts_node(node, grid, temperature=500.0):
    """Simulate a single MCTS node."""
    rng = np.random.default_rng(np.random.randint(0, 1000000))
    ttl = 5
    no_improvement = 0
    best_value = float('-inf')

    current = node
    while current.untried_actions and current.children:
        current = current.uct_select()
        current.virtual_loss += 1
        if current.value > best_value:
            best_value = current.value
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= ttl:
                break

    if current.untried_actions:
        r, c, v = current.untried_actions.pop()
        new_grid = current.grid.copy()
        new_grid[r, c] = v
        child = MCTSNode(new_grid, current, (r, c, v))
        current.children.append(child)
        current = child

    sim_result = ray.get(simulate_full_board.remote(current.grid, None, n_iter=100, seed=rng.integers(0, 1000000)))
    if not isinstance(sim_result, dict):
        logger.error(f"Invalid sim_result type: {type(sim_result)}")
        return 0.0

    reward = 0.0
    for r, c in np.argwhere(grid == -1):
        if (r, c) in sim_result:
            weighted = weight_prob_by_modules(current.grid, {(r, c): sim_result[(r, c)]})
            reward += max(weighted[(r, c)].values())

    if current.parent is not None:
        delta = reward - current.parent.value
        if delta < 0 and rng.random() > math.exp(delta / temperature):
            current.parent.children.remove(current)
            return 0.0

    while current is not None:
        current.visits += 1
        current.value += reward
        current.virtual_loss -= 1
        current = current.parent
    return reward

@ray.remote
def mcts(grid: np.ndarray, iterations: int = 1000, seed: Optional[int] = None):
    """Monte Carlo Tree Search with simulated annealing."""
    rng = np.random.default_rng(seed if seed is not None else np.random.randint(0, 1000000))
    rows, cols = grid.shape
    root = MCTSNode(grid)

    # Execute simulations in parallel
    ray.get([simulate_mcts_node.remote(root, grid) for _ in range(iterations // 2)])  # Increase parallelism
    best_child = max(root.children, key=lambda c: c.value / c.visits, default=root)
    return best_child.grid

def weight_prob_by_modules(grid: np.ndarray,
                           prob_map: Dict[Tuple[int, int], Dict[int, float]],
                           target_num: Optional[int] = None) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Weight probabilities by module scores with adaptive weighting."""
    if not isinstance(prob_map, dict):
        logger.error(f"Invalid prob_map type: {type(prob_map)}")
        return {}

    result = prob_map.copy()
    modules = select_modules(grid)
    module_scores = np.array([get_module_score(mod, grid) for mod in modules], dtype=np.float32)

    error_history = defaultdict(float)
    weights = np.array([0.4, 0.3, 0.3], dtype=np.float32)
    
    for (r, c), probs in result.items():
        if (r, c) not in prob_map:
            continue
        scores = module_scores[:, r, c]
        scores = np.nan_to_num(scores, nan=0.0)
        softmax_scores = np.exp(scores / 0.5) / (np.sum(np.exp(scores / 0.5)) + 1e-10)
        mean_score = np.mean(softmax_scores)

        if target_num is not None:
            if target_num in probs:
                probs[target_num] = max(probs[target_num] * mean_score * weights[1], 1e-10)
                total = probs[target_num] or 1e-10
                result[(r, c)] = {target_num: probs[target_num] / total}
            else:
                result[(r, c)] = {target_num: 0.0}
        else:
            for val in probs:
                probs[val] = max(probs[val] * mean_score, 1e-10)
            total = sum(probs.values()) or 1e-10
            result[(r, c)] = {k: v / total for k, v in probs.items()}

        if target_num is not None:
            error = 1.0 - probs.get(target_num, 0.0)
            error_history[(r, c)] = (error_history[(r, c)] * 4 + error) / 5
            if error_history[(r, c)] > 0.2:
                weights[1] *= 0.95
                weights = weights / (np.sum(weights) + 1e-10)

    return result

def global_unique(prob_map: Dict[Tuple[int, int], Dict[int, float]],
                  blanks: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, float]]:
    """Global unique assignment with linear sum assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
        nums = sorted({n for d in prob_map.values() for n in d})
        cost = np.full((len(blanks), len(nums)), 50.0, dtype=np.float32)

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
        for cell in sorted(blanks, key=lambda p: max(prob_map[p].values() or [0]), reverse=True):
            for n, p in sorted(prob_map[cell].items(), key=lambda x: x[1], reverse=True):
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

    def uct_select(self, c_param=1.4):
        return max(self.children,
                   key=lambda c: (c.value / (c.visits + c.virtual_loss)) +
                   c_param * np.sqrt(2 * np.log(self.visits + 1e-10) / (c.visits + c.virtual_loss + 1e-10)))

def predict_scratch_card(
    grid: List[List[int]],
    target_num: Optional[int] = None,
    iterations: Optional[int] = None,
    quick_iter: Optional[int] = None,
    refine_iter: Optional[int] = None,
    min_total_iter: Optional[int] = None,
    unique: bool = True
) -> Dict[str, Any]:
    grid_np = np.array(grid, dtype=np.int16)
    rows, cols = grid_np.shape
    blanks = [tuple(b) for b in np.argwhere(grid_np == -1)]

    if not blanks:
        return {"mode": "no_blanks", "predictions": [], "full_probabilities": {}}

    modules = [
        ("EXT_M1_Tail_Pattern_Vec", "Tail number pattern match"),
        ("EXT_M3_Local_Focus_Vec", "Neighborhood mean/variance alignment"),
        ("EXT_M10_Sequence_Block_Vec", "Sequence block continuity"),
        ("EXT_R3_Error_Correction_Vec", "Historical error correction"),
        ("EXT_F7_Strong_Pattern_Vec", "Arithmetic/symmetry pattern"),
        ("EXT_GM20_Skip_Pattern_Confidence_Vec", "Skip pattern confidence")
    ]

    base_iter = iterations if iterations is not None else int(os.getenv("ITER", 4000))
    total_iter = int(base_iter * max(rows * cols / 40, 1))
    quick_iter = quick_iter if quick_iter is not None else int(total_iter * 0.35)
    refine_iter = refine_iter if refine_iter is not None else total_iter - quick_iter
    min_total_iter = min_total_iter if min_total_iter is not None else max(1000, total_iter // 5)

    known = np.argwhere(grid_np != -1)
    known_vals = grid_np[grid_np != -1]
    best_seed, best_loss = reverse_engineer_seed(grid_np, known, known_vals, rows, cols)

    surrogate = SurrogateModel()
    coarse_seeds = np.arange(0, 100000, 10000, dtype=np.int32)
    coarse_losses = Parallel(n_jobs=4)(delayed(lambda s: reverse_engineer_seed(grid_np, known, known_vals, rows, cols, seed_range=(s, s+1))[1])(s) for s in coarse_seeds)
    surrogate.fit(coarse_seeds, coarse_losses)
    top_seeds = surrogate.select_top_seeds((0, 100000), n_samples=100)

    prob_maps = []
    for seed in top_seeds[:5]:
        prob_map = ray.get(simulate_full_board.remote(grid_np, target_num, n_iter=quick_iter, seed=seed))
        prob_maps.append(prob_map)

    final_prob_map = defaultdict(lambda: defaultdict(float))
    for prob_map in prob_maps:
        for (r, c), probs in prob_map.items():
            for num, prob in probs.items():
                final_prob_map[(r, c)][num] += prob / len(prob_maps)

    logger.info(f"Simulating full board with {total_iter} iterations, best seed: {best_seed}")
    prob_map = ray.get(simulate_full_board.remote(grid_np, target_num, n_iter=refine_iter, seed=best_seed))

    for (r, c), probs in prob_map.items():
        for num, prob in probs.items():
            final_prob_map[(r, c)][num] = 0.4 * final_prob_map[(r, c)][num] + 0.6 * prob

    if target_num is not None:
        rank = [{
            "row": r,
            "col": c,
            "candidates": [target_num],
            "probability": final_prob_map[(r, c)].get(target_num, 0.0) * 100
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
            "full_probabilities": final_prob_map
        }

    if unique:
        assign = global_unique(final_prob_map, blanks)
        best_grid = ray.get(mcts.remote(grid_np, iterations=1000, seed=best_seed))

        old_conf = max([p for (_, _), (_, p) in assign.items()] or [0])
        new_conf = max([
            max(weight_prob_by_modules(best_grid, {(r, c): final_prob_map[(r, c)]})[(r, c)].values())
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
            "full_probabilities": final_prob_map
        }

    preds = []
    for (r, c), dist in final_prob_map.items():
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
        "full_probabilities": final_prob_map
    }

def process_grid(grid):
    """Process grid for predictions."""
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