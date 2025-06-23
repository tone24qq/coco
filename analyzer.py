import os
import math
import numpy as np
import xxhash
from collections import defaultdict
from functools import lru_cache
import duckdb
import threading
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

from modules import FORMULA_REGISTRY
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

# Track how many child nodes are skipped in UCT selection
SKIPPED_NODES = 0

# 來自 probmap_key_patch_v2.txt
def _native_coord(k):
    return int(k[0]), int(k[1])

def _native_dict(d):
    return {_native_coord(k): v for k, v in d.items()}

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

def adjust_weights_based_on_history(history: Dict[str, float], formulas: Tuple[str, ...]) -> np.ndarray:
    """Dynamically adjust formula weights based on historical performance."""
    total = sum(history.get(f, 0.0) for f in formulas) or 1e-10
    return np.array([history.get(f, 0.0) / total for f in formulas])

def select_modules(grid: np.ndarray) -> List[str]:
    """Dynamically select modules based on grid characteristics."""
    # 根據 FORCE_FULL_SCAN 環境變數決定是否使用所有模組
    if os.getenv("FORCE_FULL_SCAN", "0") == "1":
        return list(REGISTERED_MODULES_BRAIN)   # 直接使用所有模組
    # 原始邏輯：動態選擇模組
    base_modules = ["EXT_Q1_ProximityEntropy_Vec", "EXT_Q2_PotentialPath_Vec", "EXT_Q5_GlobalEntropy_Vec", "EXT_Q6_LineBridge_Vec", "EXT_Q7_VariancePrior_Vec"]
    scores = {mod: np.mean(get_module_score(mod, grid)) for mod in REGISTERED_MODULES_BRAIN}
    top_modules = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:2]
    return base_modules + [m for m in top_modules if m not in base_modules]

_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'board_cache.duckdb')
_CACHE_LOCAL = threading.local()
_CACHE_INIT_LOCK = threading.Lock()
_CACHE_INITIALIZED = False

def _init_cache_conn() -> duckdb.DuckDBPyConnection:
    "Create a new DuckDB connection."
    return duckdb.connect(_CACHE_PATH)

def _ensure_cache_schema(conn: duckdb.DuckDBPyConnection) -> None:
    "Initialize table once in a thread-safe manner."
    conn.execute(
        'CREATE TABLE IF NOT EXISTS board_cache ('
        'mask TEXT, seed INTEGER, r INTEGER, c INTEGER, board BLOB,'
        'PRIMARY KEY(mask, seed, r, c))'
    )

def _get_cache_conn() -> duckdb.DuckDBPyConnection:
    """Return a valid thread-local DuckDB connection."""
    conn = getattr(_CACHE_LOCAL, 'conn', None)
    if conn is None:
        conn = _init_cache_conn()
        with _CACHE_INIT_LOCK:
            global _CACHE_INITIALIZED
            if not _CACHE_INITIALIZED:
                _ensure_cache_schema(conn)
                _CACHE_INITIALIZED = True
        _CACHE_LOCAL.conn = conn
    else:
        try:
            conn.execute('SELECT 1')
        except Exception:
            try:
                conn.close()
            finally:
                conn = _init_cache_conn()
                _CACHE_LOCAL.conn = conn
    return conn

_MEM_CACHE: Dict[Tuple[str, int, int, int], np.ndarray] = {}

def _cached_board(mask_key: str, seed: int, r: int, c: int,
                  kv_bytes: bytes, idx_bytes: bytes) -> np.ndarray:
    """Return a unique 1-D board (length r*c) with persistent caching."""
    cache_key = (mask_key, seed, r, c)
    if cache_key in _MEM_CACHE:
        return _MEM_CACHE[cache_key]

    conn = _get_cache_conn()
    try:
        row = conn.execute(
            "SELECT board FROM board_cache WHERE mask=? AND seed=? AND r=? AND c=?",
            cache_key,
        ).fetchone()
        if row:
            board = np.frombuffer(row[0], dtype=np.int16)
            if board.size == r * c:
                _MEM_CACHE[cache_key] = board
                return board
    except Exception as exc:
        logger.error("Cache read error: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        _CACHE_LOCAL.conn = _init_cache_conn()

    rng = np.random.default_rng(seed)
    n = r * c
    perm = rng.permutation(n) + 1

    idx = np.frombuffer(idx_bytes, dtype=np.int32)
    if idx.size == 0:
        board = perm.astype(np.int16)
    else:
        vals = np.frombuffer(kv_bytes, dtype=np.int32)
        if idx.size != vals.size:
            board = perm.astype(np.int16)
        else:
            mask = np.isin(perm, vals, invert=True)
            remaining = perm[mask]
            board = np.empty(n, dtype=np.int16)
            board[idx] = vals
            unknown_idx = np.setdiff1d(np.arange(n), idx, assume_unique=True)
            board[unknown_idx] = remaining[:unknown_idx.size]

    try:
        conn.execute(
            "INSERT OR REPLACE INTO board_cache VALUES (?, ?, ?, ?, ?)",
            (*cache_key, board.tobytes()),
        )
        conn.commit()
    except Exception as exc:
        logger.error("Cache write error: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        _CACHE_LOCAL.conn = _init_cache_conn()

    _MEM_CACHE[cache_key] = board
    if len(_MEM_CACHE) > 10000:
        _MEM_CACHE.pop(next(iter(_MEM_CACHE)))
    return board

def generate_full_boards(rows: int, cols: int, batch: int, rng: np.random.Generator,
                         formulas: Tuple[str, ...], weights: np.ndarray,
                         grid: np.ndarray) -> np.ndarray:
    """Generate batch of complete boards using weighted formulas with importance sampling."""
    valid = [f for f in formulas if f in FORMULA_REGISTRY]
    if not valid:
        raise ValueError("No valid formulas available")
    weights = np.array([weights[i] for i, f in enumerate(formulas) if f in FORMULA_REGISTRY], dtype=float)
    weights = weights / (weights.sum() + 1e-10)
    boards = np.empty((batch, rows, cols), dtype=np.int16)
    known_vals = grid.ravel()
    known_mask = (grid != -1).ravel()
    kv_bytes = known_vals.tobytes()
    idx_bytes = known_mask.nonzero()[0].astype(np.int32).tobytes()
    mask = xxhash.xxh64(kv_bytes + idx_bytes).hexdigest()
    seeds = rng.integers(0, 0xFFFF, size=batch)
    for i, s in enumerate(seeds):
        board1d = _cached_board(
            mask, int(s) & 0xFFFF,
            rows, cols,
            kv_bytes, idx_bytes
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
    history = {"random_entropy": 0.4, "shuffle": 0.3, "tail_cluster": 0.3}
    if np.mean(module_scores) > 0.6:
        history["tail_cluster"] += 0.1
        history["random_entropy"] -= 0.05

    formulas = ("random_entropy", "shuffle", "tail_cluster")
    weights = adjust_weights_based_on_history(history, formulas)
    remain = n_iter
    counts = defaultdict(lambda: defaultdict(int))
    prev_probs: Optional[Dict[Tuple[int, int], Dict[int, float]]] = None
    variance_map: Dict[Tuple[int, int], float] = {}

    while remain > 0:
        batch = min(4000, remain)
        boards = generate_full_boards(rows, cols, batch, rng, formulas, weights, g)

        if known.size:
            mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
            boards = boards[mask]
            if len(boards) == 0:
                batch = min(batch * 2, 8000)
                boards = generate_full_boards(rows, cols, batch, rng, formulas, weights, g)
                mask = np.all(boards[:, known[:, 0], known[:, 1]] == known_vals, axis=1)
                boards = boards[mask]

        if len(boards) > 0:
            for i, board in enumerate(boards):
                fast = 0.5 * get_module_score("EXT_Q1_ProximityEntropy_Vec", board) + 0.5 * get_module_score("EXT_Q2_PotentialPath_Vec", board) + \
                       0.2 * get_module_score("EXT_Q5_GlobalEntropy_Vec", board) + 0.2 * get_module_score("EXT_Q6_LineBridge_Vec", board) + \
                       0.1 * get_module_score("EXT_Q7_VariancePrior_Vec", board)
                for r, c in blanks:
                    idx = r * cols + c
                    if rng.random() < importance_weights[idx]:
                        num = board[r, c]
                        counts[(r, c)][num] += fast[r, c]
                        if target_num is not None and num == target_num:
                            counts[(r, c)][num] += 2 * fast[r, c]

        remain -= batch

        # --- convergence check ---
        prob_map_cur: Dict[Tuple[int, int], Dict[int, float]] = {}
        for (r, c) in [tuple(b) for b in blanks]:
            total = sum(counts[(r, c)].values()) or 1e-10
            probs = {n: counts[(r, c)][n] / total for n in legal_all}
            prob_map_cur[(r, c)] = probs
            variance_map[(r, c)] = float(np.var(list(counts[(r, c)].values()) or [0.0]))

        if prev_probs is not None:
            diff = 0.0
            denom = 0
            for cell in prob_map_cur:
                for n in legal_all:
                    diff += abs(prob_map_cur[cell].get(n, 0.0) - prev_probs[cell].get(n, 0.0))
                    denom += 1
            if denom and diff / denom < 0.002:
                break

        prev_probs = prob_map_cur

    prob_map = {}
    for (r, c) in [tuple(b) for b in blanks]:
        total = sum(counts[(r, c)].values()) or 1e-10
        probs = {n: max(counts[(r, c)][n] / total, 1e-10) for n in legal_all}
        prob_map[(r, c)] = probs

    # Two-phase scoring: Re-rank top K candidates with Borda or Soft-Max
    tau = float(os.getenv("TAU_SOFTMAX", "0.3"))
    w = np.array([0.28, 0.28, 0.12, 0.12, 0.08, 0.07, 0.05])  # Q1~Q7
    topk = int(os.getenv("TOPK_RERANK", "100"))
    if topk < 0:             # -1 表示 rows × cols
        topk = rows * cols

    candidates = [(r, c, max(probs.values()), num) for (r, c), probs in prob_map.items() for num in probs]
    top_k = heapq.nlargest(topk, candidates, key=lambda x: x[2])
    final_prob_map = {}
    for r, c, fast_score, num in top_k:
        scores = [
            get_module_score("EXT_Q1_ProximityEntropy_Vec", g)[r, c],
            get_module_score("EXT_Q2_PotentialPath_Vec", g)[r, c],
            get_module_score("EXT_Q3_DiscontinuitySym_Vec", g)[r, c],
            get_module_score("EXT_Q4_ControlComposite_Vec", g)[r, c],
            get_module_score("EXT_Q5_GlobalEntropy_Vec", g)[r, c],
            get_module_score("EXT_Q6_LineBridge_Vec", g)[r, c],
            get_module_score("EXT_Q7_VariancePrior_Vec", g)[r, c]
        ]
        soft = np.exp(np.array(scores) / tau); soft /= soft.sum() + 1e-10
        final_score = (w * soft).sum()
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

    return _native_dict(result)

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
    EPS = 1e-9  # 檔頭或 class 內自訂常數

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

    def uct_select(self):
        """Upper-Confidence bound with virtual-loss safe division"""
        def ucb(child):
            denom = child.visits + child.virtual_loss
            if denom == 0:
                return float('inf')  # 確保新節點優先被選
            exploitation = child.value / denom
            exploration = math.sqrt(2 * math.log(self.visits + 1) / denom)
            return exploitation + exploration
        scores = [(child, ucb(child)) for child in self.children]
        if not scores:
            raise ValueError("No children to select")
        global_best = max(s for _, s in scores)
        threshold = global_best * 0.95
        filtered = [child for child, s in scores if s >= threshold]
        skipped = len(scores) - len(filtered)
        if skipped:
            global SKIPPED_NODES
            SKIPPED_NODES += skipped
        return max(filtered, key=ucb)

def mcts(grid: np.ndarray, iterations: int = 1000):
    rows, cols = grid.shape
    root = MCTSNode(grid)
    global SKIPPED_NODES
    SKIPPED_NODES = 0

    def simulate(node):
        try:
            current = node
            while current.untried_actions and len(current.children) < 1.5 * current.visits ** 0.5:
                current = current.uct_select()
                current.virtual_loss += 1
            if current.untried_actions:
                r, c, v = current.untried_actions.pop()
                new_grid = current.grid.copy()
                new_grid[r, c] = v
                new_child = MCTSNode(new_grid, current, (r, c, v))
                new_child.visits = 1  # 或 new_child.visits = new_child.virtual_loss = EPS
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
                        current.grid, {(r, c): sim_result[(r, c)]})
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

    Parallel(n_jobs=4, require='sharedmem')(delayed(simulate)(root) for _ in range(iterations // 4))
    logger.info("MCTS iterations=%d | skipped_children=%d", iterations, SKIPPED_NODES)
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
    blanks = [tuple(map(int, b)) for b in np.argwhere(grid_np == -1)]  # 來自 probmap_key_patch_v2.txt

    if not blanks:
        return {"mode": "no_blanks", "predictions": [], "full_probabilities": {}}

    modules = [
        ("EXT_Q1_ProximityEntropy_Vec", "Proximity and entropy scoring"),
        ("EXT_Q2_PotentialPath_Vec", "Sequence and path scoring"),
        ("EXT_Q3_DiscontinuitySym_Vec", "Discontinuity and symmetry scoring"),
        ("EXT_Q4_ControlComposite_Vec", "Control and error correction"),
        ("EXT_Q5_GlobalEntropy_Vec", "Global entropy and clustering"),
        ("EXT_Q6_LineBridge_Vec", "Linearity and bridge connectivity"),
        ("EXT_Q7_VariancePrior_Vec", "Variance smoothing and prior")
    ]

    base_iter = iterations if iterations is not None else int(os.getenv("ITER", "5000"))
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
            "probability": prob_map.get((r, c), {}).get(target_num, 0.0) * 100
        } for r, c in prob_map.keys()]  # 改用 .get()
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
            pred["module_scores"] = {mod: float(module_scores[mod][pred['row'], pred['col']])
                                   for mod, _ in modules}

        return {
            "mode": "target",
            "target": target_num,
            "predictions": rank[:3],
            "full_probabilities": prob_map
        }

    if unique:
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
            for _, cell_probs in prob_map.items()      # 逐格取出內部 dict
            for p in cell_probs.values()               # 逐個號碼機率
        )

        # 2️⃣ 嘗試把每個候選格 (r,c) 掛回去後，重新加權 → 看能否誕生更高機率
        new_conf = max([
            max(
                weight_prob_by_modules(                # <<< 你自己已有的函式
                    best_grid,                         # - 當前最佳棋盤
                    {(r, c): prob_map.get((r, c), {})} # - 只帶單一候選格的機率
                ).get((r, c), {}).values(),            # 取回各模組加權後的機率們
                default=0                              # ← dict 為空時避免 ValueError
            )
            for (r, c) in blanks                   # 逐一測試所有候選格
        ])

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
            pred["module_scores"] = {mod: float(module_scores[mod][pred['row'], pred['col']])
                                   for mod, _ in modules}

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
        pred["module_scores"] = {mod: float(module_scores[mod][pred['row'], pred['col']])
                               for mod, _ in modules}

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