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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

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
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    return EXT_GM20_Skip_Pattern_Confidence_Vec(grid)

def simulate_batch(grid_bytes: bytes, rows: int, cols: int, batch_vec: np.ndarray, target_num: Optional[int] = None) -> Dict[Tuple[int, int], int]:
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    blanks = np.argwhere(grid == -1)
    cell_to_idx = {tuple(b): i for i, b in enumerate(blanks)}
    counts = defaultdict(int)
    
    for b in batch_vec:
        if target_num is not None:
            for r, c in blanks:
                if b[r, c] == target_num:
                    counts[(r, c)] += 1
    return counts

@lru_cache(maxsize=128)
def simulate_with_formulas(
    grid_bytes: bytes,
    rows: int,
    cols: int,
    n_iter: int = 1000,
    target_num: Optional[int] = None,
    min_iter: int = 0
) -> Dict[Tuple[int, int], Dict[int, float]]:
    grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
    blanks = np.argwhere(grid == -1)
    known_idx = np.argwhere(grid != -1)
    known_vals = grid[grid != -1]
    legal_all = analyzer_utils.get_legal_values_for_placement(grid)
    
    # Validate grid size and numbers
    if rows < 4 or rows > 20 or cols < 4 or cols > 20:
        raise ValueError("Grid must be 4x4 to 20x20")
    max_val = rows * cols
    if any(v < 1 or v > max_val for v in known_vals):
        raise ValueError(f"Numbers must be between 1 and {max_val}")
    if len(np.unique(known_vals)) != len(known_vals):
        raise ValueError("Grid contains duplicate numbers")

    # Initialize counters
    counts = defaultdict(lambda: defaultdict(int))
    rng = np.random.default_rng()
    total_seen, effective = 0, 0
    batch_size = max(500, min(4000, int(8e9 / (rows * cols * 8))))  # 8 GB RAM 調整

    # Use gen_random_entropy for whole-board shuffling
    formula = FORMULA_REGISTRY["random_entropy"]

    while total_seen < n_iter:
        need = min(batch_size, n_iter - total_seen)
        boards = np.zeros((need, rows, cols), dtype=np.int64)
        
        # Generate boards
        for i in range(need):
            board = formula(rows, cols, rng)
            # Enforce known values
            for (r, c), val in zip(known_idx, known_vals):
                if board[r, c] != val:
                    # Find and swap
                    target_pos = np.argwhere(board == val)[0]
                    board[target_pos[0], target_pos[1]], board[r, c] = board[r, c], board[target_pos[0], target_pos[1]]
            boards[i] = board
        
        # Count target_num occurrences
        results = Parallel(n_jobs=4)(
            delayed(simulate_batch)(grid_bytes, rows, cols, boards[i:i+batch_size//4], target_num)
            for i in range(0, need, batch_size//4)
        )
        
        # Aggregate counts
        for batch_counts in results:
            for (r, c), count in batch_counts.items():
                counts[(r, c)][target_num] += count
        
        total_seen += need
        effective += need
        
        # Early stopping
        if total_seen >= max(min_iter, 1000) and effective >= 1000:
            break

    # Convert counts to probabilities
    prob_map = {}
    for (r, c) in [tuple(b) for b in blanks]:
        total = sum(counts[(r, c)].values()) or 1e-10
        probs = {n: (counts[(r, c)][n] / total) for n in legal_all}
        prob_map[(r, c)] = probs
    
    # Apply module weighting
    prob_map = weight_prob_by_modules(grid, prob_map, target_num)
    
    return prob_map

def weight_prob_by_modules(grid: np.ndarray, prob_map: Dict[Tuple[int, int], Dict[int, float]], target_num: Optional[int] = None) -> Dict[Tuple[int, int], Dict[int, float]]:
    if not isinstance(prob_map, dict):
        logger.error(f"Invalid prob_map type: {type(prob_map)}")
        return {}
    
    result = prob_map.copy()
    modules = [
        "EXT_M1_Tail_Pattern_Vec",
        "EXT_M3_Local_Focus_Vec",
        "EXT_M10_Sequence_Block_Vec",
        "EXT_R3_Error_Correction_Vec",
        "EXT_F7_Strong_Pattern_Vec",
        "EXT_GM20_Skip_Pattern_Confidence_Vec"
    ]
    module_scores = Parallel(n_jobs=4)(delayed(get_module_score)(mod, grid) for mod in modules)
    module_scores = np.array(module_scores)

    for (r, c), probs in result.items():
        if (r, c) not in prob_map:
            continue
        # Compute consensus score
        scores = module_scores[:, r, c]
        if np.any(np.isnan(scores)):
            scores = np.nan_to_num(scores, nan=0.0)
        softmax_scores = np.exp(scores / 0.5) / (np.sum(np.exp(scores / 0.5)) + 1e-10)  # Temperature = 0.5
        mean_score = np.mean(softmax_scores)
        
        if target_num is not None:
            if target_num in probs:
                probs[target_num] *= mean_score
                total = probs[target_num] or 1e-10
                result[(r, c)] = {target_num: probs[target_num] / total}
            else:
                result[(r, c)] = {target_num: 0.0}
        else:
            for val in probs:
                probs[val] *= mean_score
            total = sum(probs.values()) or 1e-10
            result[(r, c)] = {k: v / total for k, v in probs.items()}
    
    return result

def global_unique(prob_map: Dict[Tuple[int, int], Dict[int, float]], blanks: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        nums = sorted({n for d in prob_map.values() for n in d})
        cost = np.full((len(blanks), len(nums)), 50.0)
        for i, cell in enumerate(blanks):
            for j, n in enumerate(nums):
                cost[i, j] = -math.log(prob_map[cell].get(n, 1e-9))
        row, col = linear_sum_assignment(cost)
        return {blanks[r]: (nums[c], prob_map[blanks[r]].get(nums[c], 0.0)) for r, c in zip(row, col)}
    except Exception as e:
        logger.error(f"Global unique assignment failed: {e}")
        assigned, res = set(), {}
        for cell in sorted(blanks, key=lambda p: max(prob_map[p].values() or [0]), reverse=True):
            for n, p in sorted(prob_map[cell].items(), key=lambda x: x[1], reverse=True):
                if n not in assigned:
                    assigned.add(n)
                    res[cell] = (n, p)
                    break
            if cell not in res:  # Fallback if no valid assignment
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
        self.untried_actions = [(r, c, v) for r, c in np.argwhere(grid == -1)
                              for v in analyzer_utils.get_legal_values_for_placement(grid)]

    def uct_select(self, c_param=1.4):
        return max(self.children, key=lambda c: c.value / c.visits + c_param * np.sqrt(2 * np.log(self.visits) / c.visits))

def mcts(grid: np.ndarray, iterations: int = 1000):
    rows, cols = grid.shape
    root = MCTSNode(grid)

    def simulate(node):
        current = node
        while current.untried_actions and current.children:
            current = current.uct_select()
        if current.untried_actions:
            r, c, v = current.untried_actions.pop()
            new_grid = current.grid.copy()
            new_grid[r, c] = v
            child = MCTSNode(new_grid, current, (r, c, v))
            current.children.append(child)
            current = child
        sim_result = simulate_with_formulas(current.grid.tobytes(), rows, cols, 500, None, 100)
        if not isinstance(sim_result, dict):
            logger.error(f"Invalid sim_result type: {type(sim_result)}")
            return 0.0
        reward = 0.0
        for r, c in np.argwhere(grid == -1):
            if (r, c) in sim_result:
                weighted_probs = weight_prob_by_modules(current.grid, {k: v for k, v in sim_result.items() if k == (r, c)})
                if weighted_probs and (r, c) in weighted_probs and weighted_probs[(r, c)]:
                    reward += max(weighted_probs[(r, c)].values())
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent
        return reward

    Parallel(n_jobs=4)(delayed(simulate)(root) for _ in range(iterations // 4))
    best_child = max(root.children, key=lambda c: c.value / c.visits, default=root)
    return best_child.grid

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

    # Dynamic iteration based on grid size
    base_iter = iterations if iterations is not None else int(os.getenv("BASE_ITER", 1000))
    total_iter = int(base_iter * max(rows * cols / 40, 1))
    quick_iter = quick_iter if quick_iter is not None else int(total_iter * 0.35)
    refine_iter = refine_iter if refine_iter is not None else total_iter - quick_iter
    min_total_iter = min_total_iter if min_total_iter is not None else max(1000, total_iter // 5)

    # Single simulation phase
    prob_map = simulate_with_formulas(
        grid_np.tobytes(), rows, cols,
        n_iter=total_iter,
        target_num=target_num,
        min_iter=min_total_iter
    )

    if target_num is not None:
        # Rank cells for target number
        rank = [{
            "row": r,
            "col": c,
            "candidates": [target_num],
            "probability": prob_map[(r, c)].get(target_num, 0.0) * 100  # Convert to percentage
        } for r, c in blanks]
        rank.sort(key=lambda x: x["probability"], reverse=True)
        
        # Add reasons for Top-3
        modules = [
            ("EXT_M1_Tail_Pattern_Vec", "Tail number pattern match"),
            ("EXT_M3_Local_Focus_Vec", "Neighborhood mean/variance alignment"),
            ("EXT_M10_Sequence_Block_Vec", "Sequence block continuity"),
            ("EXT_R3_Error_Correction_Vec", "Historical error correction"),
            ("EXT_F7_Strong_Pattern_Vec", "Arithmetic/symmetry pattern"),
            ("EXT_GM20_Skip_Pattern_Confidence_Vec", "Skip pattern confidence")
        ]
        module_scores = {mod: get_module_score(mod, grid_np) for mod, _ in modules}
        
        for pred in rank[:3]:
            reasons = []
            scores = [(mod, module_scores[mod][pred['row'], pred['col']], desc) for mod, desc in modules]
            top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for mod, score, desc in top_modules:
                if score > 0.5:  # Threshold for significant contribution
                    reasons.append(f"{desc} (score: {score:.2f})")
            pred["reasons"] = reasons if reasons else ["No dominant module contribution"]
        
        return {
            "mode": "target",
            "target": target_num,
            "predictions": rank[:3],
            "full_probabilities": prob_map
        }

    # General case (no target number)
    if unique:
        assign = global_unique(prob_map, blanks)
        best_grid = mcts(grid_np, iterations=1000)
        old_conf = max([p for (_, _), (_, p) in assign.items()] or [0])
        new_conf = max([max(weight_prob_by_modules(best_grid, prob_map[(r, c)]).values() or [0]) for r, c in blanks])
        preds = [{
            "row": r,
            "col": c,
            "candidates": [n],
            "probability": float(p) * 100  # Convert to percentage
        } for (r, c), (n, p) in (assign.items() if new_conf <= old_conf * 0.95 else process_grid(best_grid))]
        preds.sort(key=lambda x: x["probability"], reverse=True)
        
        # Add reasons
        module_scores = {mod: get_module_score(mod, grid_np) for mod, desc in modules}
        for pred in preds[:3]:
            reasons = []
            scores = [(mod, module_scores[mod][pred['row'], pred['col']], desc) for mod, desc in modules]
            top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            for mod, score, desc in top_modules:
                if score > 0.5:
                    reasons.append(f"{desc} (score: {score:.2f})")
            pred["reasons"] = reasons if reasons else ["No dominant module contribution"]
        
        return {
            "mode": "mcts_unique" if new_conf > old_conf * 0.95 else "unique",
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
            "probability": list(map(lambda x: x * 100, probs))  # Convert to percentage
        })
    
    # Add reasons
    module_scores = {mod: get_module_score(mod, grid_np) for mod, desc in modules}
    for pred in preds[:3]:
        reasons = []
        scores = [(mod, module_scores[mod][pred['row'], pred['col']], desc) for mod, desc in modules]
        top_modules = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        for mod, score, desc in top_modules:
            if score > 0.5:
                reasons.append(f"{desc} (score: {score:.2f})")
        pred["reasons"] = reasons if reasons else ["No dominant module contribution"]
    
    preds.sort(key=lambda x: x["probability"][0] if x["probability"] else 0, reverse=True)
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
        max_prob = max(legal_vals) if legal_vals else 1  # Fallback to 1 if empty
        preds.append({
            "row": r,
            "col": c,
            "candidates": [max_prob],
            "probability": 100.0 if grid[r, c] != -1 else 50.0  # Convert to percentage
        })
    return preds