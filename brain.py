    import math
    import logging
    from collections import Counter, defaultdict
    from typing import List, Tuple, Callable, Optional, Dict, Any
    import numpy as np
    import random
    from numba import njit
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    # Math helpers
    class MathUtils:
        """Utility functions for common mathematical operations."""
    
        @staticmethod
        @njit
        def sigmoid(x: float, k: float = 1.0) -> float:
            """Clamped sigmoid to avoid overflow."""
            clamped_x = max(-700.0, min(700.0, -k * x))
            return 1 / (1 + math.exp(clamped_x)) if not math.isinf(clamped_x) else (0.0 if -k * x > 0 else 1.0)
    
        @staticmethod
        @njit
        def isclose(a: float, b: float, rel_tol: float = 1e-9) -> bool:
            """Numba-safe isclose implementation."""
            return abs(a - b) <= rel_tol * max(abs(a), abs(b))
    
        @staticmethod
        @njit
        def normalize_value(value: float, min_val: float, max_val: float, clamp: bool = True) -> float:
            """Normalize value to [0, 1]."""
            if MathUtils.isclose(max_val, min_val, 1e-9):
                return 0.5 if MathUtils.isclose(value, min_val, 1e-9) else (0.0 if value < min_val else 1.0)
            normalized = (value - min_val) / (max_val - min_val + 1e-10)
            return max(0.0, min(1.0, normalized)) if clamp else normalized
    
        @staticmethod
        def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
            """Compute Manhattan distance between two (row, col) points."""
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    # Board analysis helpers
    class BoardAnalyzerUtils:
        """Utility collection for scratch-card grid analysis."""
        @staticmethod
        @njit
        def get_neighborhood_values(
            grid: np.ndarray,
            r: int,
            c: int,
            radius: int = 2,
            eight_connectivity: bool = True,
            include_center: bool = False,
        ) -> List[float]:
            """Collect values surrounding grid[r, c] in a square radius."""
            neighbors = []
            rows, cols = grid.shape
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if not include_center and dr == 0 and dc == 0:
                        continue
                    if not eight_connectivity and abs(dr) + abs(dc) > radius:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                        neighbors.append(float(grid[nr, nc]))
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
                lambda r, c: [(r+i, c) for i in range(min_len)],
                lambda r, c: [(r, c+i) for i in range(min_len)],
                lambda r, c: [(r+i, c+i) for i in range(min_len)],
                lambda r, c: [(r+i, c-i) for i in range(min_len)],
                lambda r, c: [(r, c), (r+1, c+2), (r+2, c+1)],
                lambda r, c: [(r+i, c) for i in range(2)] + [(r+2, c+2)],
            ]
    
            for r in range(rows):
                for c in range(cols):
                    if board[r, c] == -1:
                        continue
                    for shape_gen in shapes:
                        points = [(rr, cc) for rr, cc in shape_gen(r, c) if 0 <= rr < rows and 0 <= cc < cols]
                        if len(points) >= min_len:
                            values = [board[rr, cc] for rr, cc in points if board[rr, cc] != -1]
                            if len(values) >= min_len and self.get_arithmetic_or_geometric_sequences(np.array(values), min_len, allow_gaps):
                                return True
            return False
    
        @staticmethod
        @njit
        def get_arithmetic_or_geometric_sequences(
            line: np.ndarray,
            min_len: int = 3,
            allow_gaps: int = 1,
        ) -> List[List[int]]:
            """Detect arithmetic/geometric subsequences in a 1-D array."""
            sequences = []
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
                                        sequences.append(seq_vals.copy())
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
                            sequences.append(seq_vals.copy())
            return sequences
    
        def get_card_max_value_from_gridDimensions(self, grid_shape: Tuple[int, int]) -> int:
            """Return rows×cols (max possible face value)."""
            rows, cols = grid_shape
            return rows * cols if rows and cols else 0
    
        def get_legal_values_for_placement(self, grid: np.ndarray) -> set[int]:
            """Return unused numbers > 0 that can still appear on the board."""
            rows, cols = grid.shape
            all_vals = set(range(1, self.get_card_max_value_from_gridDimensions((rows, cols)) + 1))
            used = set(int(v) for v in grid.flatten() if v != -1 and v > 0)
            return all_vals - used
    
    # Module registry
    REGISTERED_MODULES_BRAIN: Dict[str, Callable[[np.ndarray, Optional[str]], np.ndarray]] = {}
    
    def get_module_score(module_name: str, grid: np.ndarray, **kwargs) -> np.ndarray:
        """Retrieve and execute a specific scoring module from the registry."""
        effective_request_id = kwargs.get("request_id", "N/A")
        if module_name not in REGISTERED_MODULES_BRAIN:
            logger.error(f"Module {module_name} not found in REGISTERED_MODULES_BRAIN.", extra={"request_id": effective_request_id})
            rows, cols = grid.shape
            return np.zeros((rows, cols), dtype=np.float32)
        module_func = REGISTERED_MODULES_BRAIN[module_name]
        logger.info(f"Executing module: {module_name}", extra={"request_id": effective_request_id})
        try:
            score_grid = module_func(grid, **kwargs)
            return score_grid
        except Exception as e:
            logger.error(f"Error executing module {module_name}: {e}", exc_info=True, extra={"request_id": effective_request_id})
            rows, cols = grid.shape
            return np.zeros((rows, cols), dtype=np.float32)
    
    # Scoring modules
    def EXT_M1_Tail_Pattern_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
        """Score based on tail number patterns in 5x5 neighborhood."""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        utils = BoardAnalyzerUtils()
        radius = min(2, min(rows, cols) // 2 - 1)
    
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != -1:
                    continue
                neighbors = utils.get_neighborhood_values(grid, r, c, radius=radius, eight_connectivity=True)
                if not neighbors:
                    continue
                tails = np.array([int(v % 10) for v in neighbors if v > 0], dtype=np.int16)
                tail_counts = np.bincount(tails, minlength=10)
                total_tails = np.sum(tail_counts) or 1e-10
                legal_values = utils.get_legal_values_for_placement(grid)
                max_score = 0.0
                mean_val = np.mean([v for v in grid[grid != -1] if v > 0]) or 1.0
                for val in legal_values:
                    tail = val % 10
                    base_score = tail_counts[tail] / total_tails
                    distance_factor = 1.0 - (abs(val - mean_val) % 10) * 0.05
                    score = base_score * distance_factor + np.random.uniform(0, 0.1)
                    max_score = max(max_score, MathUtils().normalize_value(score, 0, 1.0))
                scores[r, c] = max_score
        return scores
    
    def EXT_M3_Local_Focus_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
        """Score based on 5x5 neighborhood mean and variance."""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        utils = BoardAnalyzerUtils()
        radius = min(2, min(rows, cols) // 2 - 1)
    
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != -1:
                    continue
                neighbors = utils.get_neighborhood_values(grid, r, c, radius=radius, eight_connectivity=True)
                if len(neighbors) < 2:
                    continue
                mean_val = np.mean(neighbors)
                std_val = np.std(neighbors, ddof=1) or 1.0
                row_seq = utils.check_sequences(grid[max(0, r-2):min(rows, r+3)], grid, min_len=3, allow_gaps=1)
                col_seq = utils.check_sequences(grid[:, max(0, c-2):min(cols, c+3)].T, grid, min_len=3, allow_gaps=1)
                legal_values = utils.get_legal_values_for_placement(grid)
                max_score = 0.0
                for val in legal_values:
                    deviation = abs(val - mean_val) / std_val
                    seq_bonus = 0.3 if (row_seq or col_seq) and abs(val - mean_val) > std_val else 0.0
                    score = MathUtils().normalize_value(deviation + seq_bonus, 0, max(1.0, std_val + 0.3))
                    max_score = max(max_score, score)
                scores[r, c] = max_score
        return scores
    
    def EXT_M10_Sequence_Block_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
        """Score based on sequence blocks in 5x5 neighborhood."""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        utils = BoardAnalyzerUtils()
        radius = min(2, min(rows, cols) // 2 - 1)
    
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != -1:
                    continue
                row_block = grid[max(0, r-2):min(rows, r+3)]
                col_block = grid[:, max(0, c-2):min(cols, c+3)]
                row_seqs = []
                for i in range(row_block.shape[0]):
                    row_seqs.extend(utils.get_arithmetic_or_geometric_sequences(row_block[i]))
                col_seqs = []
                for i in range(col_block.shape[1]):
                    col_seqs.extend(utils.get_arithmetic_or_geometric_sequences(col_block[:, i]))
                diag_seqs = []
                sub_grid = grid[max(0, r-2):min(rows, r+3), max(0, c-2):min(cols, c+3)]
                for offset in range(-min(sub_grid.shape[0], sub_grid.shape[1]), min(sub_grid.shape[0], sub_grid.shape[1])):
                    diag = np.diagonal(sub_grid, offset)
                    if len(diag) >= min(3, len(diag)):
                        diag_seqs.extend(utils.get_arithmetic_or_geometric_sequences(diag))
                    diag_flipped = np.diagonal(np.fliplr(sub_grid), offset)
                    if len(diag_flipped) >= min(3, len(diag_flipped)):
                        diag_seqs.extend(utils.get_arithmetic_or_geometric_sequences(diag_flipped))
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
    
    def EXT_R3_Error_Correction_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
        """Score based on historical error correction in 5x5 neighborhood."""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        utils = BoardAnalyzerUtils()
        radius = min(2, min(rows, cols) // 2 - 1)
    
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != -1:
                    continue
                neighbors = utils.get_neighborhood_values(grid, r, c, radius=radius, eight_connectivity=True)
                base_score = 0.5
                legal_values = utils.get_legal_values_for_placement(grid)
                for val in legal_values:
                    error_count = 0.0
                    for nr, nc in [(r+dr, c+dc) for dr in range(-radius, radius+1) for dc in range(-radius, radius+1) if 0 <= r+dr < rows and 0 <= c+dc < cols]:
                        error_count += 0.1
                    penalty = min(0.3, error_count * 0.05)
                    score = MathUtils().normalize_value(base_score - penalty, 0, 1.0)
                    if score > scores[r, c]:
                        scores[r, c] = score
        return scores
    
    def EXT_F7_Strong_Pattern_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
        """Score based on strong arithmetic or symmetry patterns."""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        utils = BoardAnalyzerUtils()
    
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != -1:
                    continue
                row_seq = utils.check_sequences(grid[r:r+1], grid, min_len=3, allow_gaps=1)
                col_seq = utils.check_sequences(grid[:, c:c+1].T, grid, min_len=3, allow_gaps=1)
                symmetry = (r == cols - 1 - c or c == rows - 1 - r)
                legal_values = utils.get_legal_values_for_placement(grid)
                max_score = 0.0
                for val in legal_values:
                    base_score = 0.5
                    if row_seq or col_seq:
                        base_score += 0.3
                    if symmetry and (0 <= rows-1-r < rows and 0 <= cols-1-c < cols) and grid[rows-1-r, cols-1-c] == val:
                        base_score += 0.2
                    score = MathUtils().normalize_value(base_score, 0, 1.0)
                    max_score = max(max_score, score)
                scores[r, c] = max_score
        return scores
    
    def EXT_GM20_Skip_Pattern_Confidence_Vec(grid: np.ndarray, request_id: Optional[str] = "N/A") -> np.ndarray:
        """Score based on skip pattern confidence."""
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        revealed = [
            {"value": int(grid[r, c]), "r": r, "c": c}
            for r in range(rows)
            for c in range(cols)
            if grid[r, c] != -1 and grid[r, c] > 0
        ]
        if not revealed:
            return scores
    
        utils = BoardAnalyzerUtils()
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
    
        skip_vecs_list = np.array(list(skip_vecs.values()), dtype=np.int16)
        unique_vecs, counts = np.unique(skip_vecs_list, axis=0, return_counts=True)
        min_occ = max(1, int(len(skip_vecs) * 0.05))
        dominant_patterns = []
        for vec, cnt in zip(unique_vecs, counts):
            if cnt < min_occ:
                break
            pattern_vals = np.array([v for v, sv in skip_vecs.items() if np.array_equal(sv, vec)], dtype=np.int16)
            pattern_vals = np.sort(pattern_vals)
            strength = MathUtils.normalize_value(cnt, min_occ, len(skip_vecs)) * 1.1
            dominant_patterns.append({"skip": tuple(vec), "values": pattern_vals, "strength": strength})
    
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
                                seq = np.sort(np.append(pat["values"], num))
                                if len(seq) >= 2 and len(np.unique(np.diff(seq))) == 1:
                                    enh += 0.5
                                elif len(seq) >= 3 and min(seq) < num < max(seq):
                                    enh += 0.15
                            best_conf = max(best_conf, pat["strength"] * enh)
                scores[r, c] = MathUtils.normalize_value(best_conf, 0, 1.0)
        return scores
    
        skip_vecs_list = np.array(list(skip_vecs.values()), dtype=np.int16)
        unique_vecs, counts = np.unique(skip_vecs_list, axis=0, return_counts=True)
        min_occ = max(1, int(len(skip_vecs) * 0.05))
        dominant_patterns = []
        for vec, cnt in zip(unique_vecs, counts):
            if cnt < min_occ:
                break
            pattern_vals = np.array([v for v, sv in skip_vecs.items() if np.array_equal(sv, vec)], dtype=np.int16)
            pattern_vals = np.sort(pattern_vals)
            strength = math_utils.normalize_value(cnt, min_occ, len(skip_vecs)) * 1.1
            dominant_patterns.append({"skip": tuple(vec), "values": pattern_vals, "strength": strength})
    
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
                                seq = np.sort(np.append(pat["values"], num))
                                if len(seq) >= 2 and len(np.unique(np.diff(seq))) == 1:
                                    enh += 0.5
                                elif len(seq) >= 3 and min(seq) < num < max(seq):
                                    enh += 0.15
                            best_conf = max(best_conf, pat["strength"] * enh)
                scores[r, c] = math_utils.normalize_value(best_conf, 0, 1.0)
        return scores
    
    # Register modules
    REGISTERED_MODULES_BRAIN.update({
        "EXT_M1_Tail_Pattern_Vec": EXT_M1_Tail_Pattern_Vec,
        "EXT_M3_Local_Focus_Vec": EXT_M3_Local_Focus_Vec,
        "EXT_M10_Sequence_Block_Vec": EXT_M10_Sequence_Block_Vec,
        "EXT_R3_Error_Correction_Vec": EXT_R3_Error_Correction_Vec,
        "EXT_F7_Strong_Pattern_Vec": EXT_F7_Strong_Pattern_Vec,
        "EXT_GM20_Skip_Pattern_Confidence_Vec": EXT_GM20_Skip_Pattern_Confidence_Vec,
    })
    
    # Verification
    if __name__ == "__main__":
        print("Verifying brain.py structure...")
        dummy_grid = np.array([[1, 2, -1], [-1, 1, 5], [3, -1, 4]], dtype=np.int16)
        print(f"Created dummy grid:\n{dummy_grid}")
        for module_to_test in ["EXT_M1_Tail_Pattern_Vec", "EXT_M3_Local_Focus_Vec", "EXT_M10_Sequence_Block_Vec"]:
            print(f"Testing get_module_score with '{module_to_test}'...")
            try:
                scores = get_module_score(module_to_test, dummy_grid)
                print(f"Successfully called {module_to_test}. Output:\n{scores}")
                assert isinstance(scores, np.ndarray), "Return type is not np.ndarray"
                assert scores.shape == dummy_grid.shape, "Return shape does not match grid shape"
                assert scores.dtype == np.float32, "Return dtype is not float32"
            except ValueError as e:
                print(f"Error: {e}")
        print("Listing all registered modules:")
        for i, name in enumerate(REGISTERED_MODULES_BRAIN.keys()):
            print(f" {i+1}. {name}")
        print(f"Total modules registered: {len(REGISTERED_MODULES_BRAIN)}")
        print("brain.py verification complete.")