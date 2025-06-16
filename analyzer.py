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
            self.w, self.d = width, depth
            self.table = np.zeros((depth, width), dtype=np.uint32)
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
    
    # Precompute skip scores
    @lru_cache(maxsize=1024)
    def precompute_skip_scores(grid_bytes: bytes, rows: int, cols: int) -> np.ndarray:
        grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
        return EXT_GM20_Skip_Pattern_Confidence_Vec(grid)
    
    def simulate_batch(grid_bytes: bytes, rows: int, cols: int, batch_vec: np.ndarray, quick_mode: bool, skip_scores: Optional[np.ndarray]) -> np.ndarray:
        grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
        blanks = np.argwhere(grid == -1)
        legal_all = analyzer_utils.get_legal_values_for_placement(grid)
        cms = CountMinSketch()
        cell_to_idx = {tuple(b): i for i, b in enumerate(blanks)}
        valid_boards = batch_vec
    
        if not quick_mode and skip_scores is not None:
            corrs = []
            for b in valid_boards:
                b_scores = EXT_GM20_Skip_Pattern_Confidence_Vec(b)
                if np.any(np.isnan(b_scores)) or np.any(np.isnan(skip_scores)):
                    corrs.append(0.0)
                else:
                    corr = np.corrcoef(skip_scores.ravel(), b_scores.ravel())[0, 1]
                    corrs.append(corr if not np.isnan(corr) else 0.0)
            valid_mask = np.array(corrs) > 0.85
            valid_boards = valid_boards[valid_mask]
    
        for b in valid_boards:
            for r, c in blanks:
                cms.update(pack_key(cell_to_idx[(r, c)], int(b[r, c])))
        return cms.table
    
    @lru_cache(maxsize=128)
    def simulate_with_formulas(
        grid_bytes: bytes,
        rows: int,
        cols: int,
        n_iter: int,
        quick_mode: bool = False,
        min_iter: int = 0
    ) -> Dict[Tuple[int, int], Dict[int, float]]:
        grid = np.frombuffer(grid_bytes, dtype=np.int64).reshape(rows, cols)
        blanks = np.argwhere(grid == -1)
        known_idx = np.argwhere(grid != -1)
        known_vals = grid[grid != -1]
        lin_known = rows * known_idx[:, 0] + known_idx[:, 1]
    
        cms = CountMinSketch()
        cell_to_idx = {tuple(b): i for i, b in enumerate(blanks)}
        legal_all = analyzer_utils.get_legal_values_for_placement(grid)
    
        skip_scores = precompute_skip_scores(grid_bytes, rows, cols) if not quick_mode else None
        mean_val, std_val = compute_global_features(grid.astype(np.float32))[:2]
        std_val = std_val or 1.0
    
        batch_size = max(500, 10000 // (rows * cols))
        total_seen, effective = 0, 0
    
        sobol_dim = rows * cols
        qmc_engine = qmc.Sobol(d=sobol_dim, scramble=True)
        bits = math.ceil(math.log2(batch_size))
        step_vecs = 2 ** bits
    
        while total_seen < n_iter:
            need = min(step_vecs, n_iter - total_seen)
            # Adjust need to nearest power of 2 or use random if needed
            need_power2 = 2 ** math.ceil(math.log2(need)) if need > 0 else 1
            vec = qmc_engine.random(need_power2) if need_power2 != step_vecs else qmc_engine.random_base2(int(math.log2(need_power2)))
            boards = (vec[:need] * (rows * cols)).astype(np.int64).reshape(-1, rows, cols)
    
            valid_mask = np.all(boards.reshape(-1, rows * cols)[:, lin_known] == known_vals, axis=1)
            if valid_mask.any():
                valid_boards = boards[valid_mask]
                results = Parallel(n_jobs=-1)(
                    delayed(simulate_batch)(grid_bytes, rows, cols, valid_boards[i:i+batch_size//4], quick_mode, skip_scores)
                    for i in range(0, len(valid_boards), batch_size//4)
                )
                for table in results:
                    for i in range(cms.d):
                        cms.table[i] += table[i]
    
            total_seen += need
            effective += valid_mask.sum()
    
            if total_seen >= max(min_iter, 3000) and not quick_mode and effective >= 3000:
                converged = True
                for (r, c), idx in cell_to_idx.items():
                    counts = [cms.query(pack_key(idx, n)) for n in legal_all]
                    s = sum(counts)
                    if s == 0 or max(counts) / s <= 0.88:
                        converged = False
                        break
                if converged:
                    break
    
        prob_map = {}
        for (r, c) in [tuple(b) for b in blanks]:
            idx = cell_to_idx[(r, c)]
            cnts = {n: cms.query(pack_key(idx, n)) for n in legal_all}
            if not any(cnts.values()):
                # Default uniform distribution for missing cells
                probs = {n: 1.0 / len(legal_all) for n in legal_all}
            else:
                v_min, v_max = min(cnts.values()), max(cnts.values())
                probs = {k: math_utils.normalize_value(v, v_min or 1e-10, v_max or 1e-10) for k, v in cnts.items()}
            prob_map[(r, c)] = probs
        return prob_map
    
    def weight_prob_by_modules(grid: np.ndarray, prob_map: Dict[Tuple[int, int], Dict[int, float]]) -> Dict[Tuple[int, int], Dict[int, float]]:
        result = prob_map.copy()
        modules = [
            "EXT_M1_Tail_Pattern_Vec",
            "EXT_M3_Local_Focus_Vec",
            "EXT_M10_Sequence_Block_Vec",
            "EXT_R3_Error_Correction_Vec",
            "EXT_F7_Strong_Pattern_Vec",
            "EXT_GM20_Skip_Pattern_Confidence_Vec"
        ]
        for (r, c), probs in result.items():
            if (r, c) not in prob_map:
                continue
            module_scores = np.zeros(len(modules))
            for i, mod in enumerate(modules):
                score_grid = get_module_score(mod, grid)
                module_scores[i] = score_grid[r, c] if score_grid.shape == grid.shape else 0.0
            weight = np.mean(module_scores) / (np.max(module_scores) or 1e-10)
            for val, prob in probs.items():
                probs[val] *= weight
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
        except Exception:
            assigned, res = set(), {}
            for cell in sorted(blanks, key=lambda p: max(prob_map[p].values()), reverse=True):
                for n, p in sorted(prob_map[cell].items(), key=lambda x: x[1], reverse=True):
                    if n not in assigned:
                        assigned.add(n)
                        res[cell] = (n, p)
                        break
            return res
    
    def predict_scratch_card(
        grid: List[List[int]],
        target_num: Optional[int] = None,
        quick_iter: Optional[int] = None,
        refine_iter: Optional[int] = None,
        min_total_iter: Optional[int] = None,
        unique: bool = True
    ) -> Dict[str, Any]:
        grid_np = np.array(grid, dtype=np.int64)
        rows, cols = grid_np.shape
        h, w = rows, cols
        blanks = [tuple(b) for b in np.argwhere(grid_np == -1)]
    
        if not blanks:
            return {"mode": "no_blanks", "predictions": [], "full_probabilities": {}}
    
        # Dynamic iteration based on grid size
        base_iter = int(os.getenv("BASE_ITER", 50000))
        total_iter = int(base_iter * max(h * w / 40, 1))
        quick_iter = quick_iter if quick_iter is not None else int(total_iter * 0.35)
        refine_iter = refine_iter if refine_iter is not None else total_iter - quick_iter
        min_total_iter = min_total_iter if min_total_iter is not None else max(10000, total_iter // 5)
    
        quick = simulate_with_formulas(
            grid_np.tobytes(), rows, cols,
            n_iter=quick_iter,
            quick_mode=True,
            min_iter=min_total_iter // 5
        )
        quick = weight_prob_by_modules(grid_np, quick)
    
        hot = sorted(blanks, key=lambda p: max(quick[p].values()), reverse=True)[:min(3, len(blanks))]
    
        refine = simulate_with_formulas(
            grid_np.tobytes(), rows, cols,
            n_iter=refine_iter,
            quick_mode=False,
            min_iter=min_total_iter - quick_iter
        )
        refine = weight_prob_by_modules(grid_np, refine)
    
        final_map = {}
        for cell in blanks:
            if cell in hot and cell in refine:
                final_map[cell] = refine[cell]
            elif cell in quick:
                final_map[cell] = quick[cell]
            else:
                legal_all = analyzer_utils.get_legal_values_for_placement(grid_np)
                final_map[cell] = {n: 1.0 / len(legal_all) for n in legal_all}
    
        if unique and target_num is None:
            assign = global_unique(final_map, blanks)
            preds = [{
                "row": r, "col": c,
                "candidates": [n], "confidences": [float(p)]
            } for (r, c), (n, p) in assign.items()]
            preds.sort(key=lambda x: x["confidences"][0], reverse=True)
            return {"mode": "unique", "predictions": preds, "full_probabilities": final_map}
    
        if target_num is not None:
            rank = [{
                "row": r, "col": c,
                "candidate": target_num,
                "confidence": final_map[(r, c)].get(target_num, 0.0)
            } for r, c in blanks]
            rank.sort(key=lambda x: x["confidence"], reverse=True)
            return {"target": target_num, "rankings": rank, "full_probabilities": final_map}
    
        preds = []
        for (r, c), dist in final_map.items():
            top3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
            nums, conf = zip(*top3)
            preds.append({
                "row": r, "col": c,
                "candidates": list(nums),
                "confidences": list(map(float, conf))
            })
        preds.sort(key=lambda x: x["confidences"][0], reverse=True)
        return {"mode": "top3", "predictions": preds, "full_probabilities": final_map}