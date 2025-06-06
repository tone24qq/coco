# vectorized_brain_modules.py

import numpy as np
from scipy.ndimage import convolve, zoom
from scipy.signal import convolve2d
import numba as nb
import logging
import json
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def scoring_module(fn):
    """Decorator to mark a method as a scoring module."""
    fn.is_scoring = True
    return fn

class VectorizedBrainModules:
    """Vectorized analyzer with multiple scoring modules for grid analysis."""
    
    def __init__(self):
        """Initialize with cached heatmaps from JSON samples."""
        self.heatmap_cache = {}  # Dict to store heatmaps by size
        self._load_heatmap()
        self.module_list = [method for method in self.__class__.__dict__.values() 
                           if callable(method) and getattr(method, 'is_scoring', False)]
        logger.info(f"Initialized VectorizedBrainModules with {len(self.module_list)} scoring modules")
        logger.debug(f"Heatmap cache contains {len(self.heatmap_cache)} sizes: {list(self.heatmap_cache.keys())}")

    def _load_heatmap(self) -> None:
        """Load JSON samples and build normalized heatmaps for each grid size."""
        try:
            samples_dir = Path(__file__).parent / "samples" / "data"
            logger.info(f"[DEBUG] Looking for JSON under: {samples_dir}")
            logger.info(f"[DEBUG] samples_dir.exists() = {samples_dir.exists()}")
            logger.info(f"[DEBUG] samples_dir.is_dir() = {samples_dir.is_dir()}")
            if not (samples_dir.exists() and samples_dir.is_dir()):
                logger.warning(f"[WARN] samples/data directory does not exist or is not a directory")
                self.heatmap_cache = {}
                return

            json_files = list(samples_dir.glob("*.json"))
            logger.info(f"[DEBUG] Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
            if not json_files:
                logger.warning(f"No JSON files found in {samples_dir}")
                self.heatmap_cache = {}
                return

            # Group files by grid size
            size_to_files = {}
            for json_file in json_files:
                try:
                    data = json.loads(json_file.read_text(encoding='utf-8'))
                    if not isinstance(data.get("grid"), list) or not data.get("grid"):
                        logger.warning(f"[WARN] Invalid or empty grid in {json_file.name}, skipping")
                        continue
                    rows, cols = len(data["grid"]), len(data["grid"][0])
                    if not all(len(row) == cols for row in data["grid"]):
                        logger.warning(f"[WARN] Inconsistent row lengths in {json_file.name}, skipping")
                        continue
                    size_key = (rows, cols)
                    size_to_files.setdefault(size_key, []).append((json_file, data))
                except Exception as e:
                    logger.warning(f"[WARN] Failed to parse {json_file.name}: {e}, skipping")

            # Create heatmap for each size
            for size_key, files in size_to_files.items():
                rows, cols = size_key
                heatmap = np.zeros((rows, cols), dtype=np.int32)
                valid_files = 0

                for json_file, data in files:
                    try:
                        ans = data.get("answer")
                        if ans is None:
                            logger.warning(f"[WARN] Answer is None in {json_file.name}, skipping")
                            continue
                        if isinstance(ans, dict) and "row" in ans and "col" in ans:
                            try:
                                r, c = int(ans["row"]) - 1, int(ans["col"]) - 1  # 1-based to 0-based
                            except (TypeError, ValueError):
                                logger.warning(f"[WARN] Invalid row/col type in {json_file.name}: {ans}, skipping")
                                continue
                        elif isinstance(ans, list) and len(ans) == 2:
                            try:
                                r, c = int(ans[0]) - 1, int(ans[1]) - 1  # Support list format
                            except (TypeError, ValueError):
                                logger.warning(f"[WARN] Invalid list values in {json_file.name}: {ans}, skipping")
                                continue
                        else:
                            logger.warning(f"[WARN] Invalid answer format in {json_file.name}: {ans}, skipping")
                            continue

                        if 0 <= r < rows and 0 <= c < cols:
                            heatmap[r, c] += 1
                            valid_files += 1
                        else:
                            logger.warning(f"[WARN] Answer out of range in {json_file.name}: row={r+1}, col={c+1}")
                    except Exception as e:
                        logger.warning(f"[WARN] Failed to process {json_file.name}: {e}, skipping")

                # Normalize heatmap
                min_val, max_val = heatmap.min(), heatmap.max()
                if max_val > min_val:
                    self.heatmap_cache[size_key] = ((heatmap - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
                else:
                    self.heatmap_cache[size_key] = np.zeros_like(heatmap, dtype=np.float32)
                logger.info(f"Loaded heatmap for size {size_key} from {valid_files} valid JSON samples, shape=({rows}, {cols})")

            if not self.heatmap_cache:
                logger.warning(f"[WARN] No valid heatmaps loaded")
        except Exception as e:
            logger.error(f"[ERROR] Heatmap loading failed: {e}")
            self.heatmap_cache = {}

    def _connectivity_heatmap_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute connectivity heatmap scores (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        
        if not self.heatmap_cache:
            logger.warning(f"[WARN] No valid heatmap available, using uniform scores")
            return np.where(legal_mask, 0.5, 0).astype(np.float32)
        
        # Find closest heatmap size
        target_size = (rows, cols)
        if target_size in self.heatmap_cache:
            heatmap = self.heatmap_cache[target_size]
            logger.debug(f"[DEBUG] Using exact heatmap match for size {target_size}")
        else:
            # Choose closest size by minimizing area difference
            size_diffs = [(size, abs(size[0] * size[1] - rows * cols)) for size in self.heatmap_cache]
            closest_size = min(size_diffs, key=lambda x: x[1])[0]
            heatmap = self.heatmap_cache[closest_size]
            logger.debug(f"[DEBUG] Using closest heatmap size {closest_size} for grid size {target_size}")
        
        # Resize heatmap to match grid
        zoom_factors = (rows / heatmap.shape[0], cols / heatmap.shape[1])
        resized_heatmap = zoom(heatmap, zoom_factors, order=1)
        scores = np.where(legal_mask, resized_heatmap, 0)
        return (scores / (np.max(scores + 1e-8) or 1.0)).astype(np.float32)

    def _edge_proximity_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute edge proximity scores (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        filled_mask = (grid > 0).astype(np.float32)
        
        radius = 2
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        distances = np.abs(x) + np.abs(y)
        distances[radius, radius] = np.inf
        kernel = 1.0 / (distances ** 1.5 + 1e-8)
        kernel[radius, radius] = 0
        
        value_weights = np.where(filled_mask, grid * 0.1, 0)
        scores = convolve(value_weights, kernel, mode='constant')
        
        max_val = np.max(scores[legal_mask]) if np.any(legal_mask) else 1.0
        return np.where(legal_mask, scores / (max_val or 1.0), 0).astype(np.float32)

    @scoring_module
    def edge_proximity_fusion(self, grid: np.ndarray) -> np.ndarray:
        """Module 1: Edge proximity fusion, low values near edges."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for edge_proximity_fusion")
            return np.zeros((1, 1), dtype=np.float32)
        return self._edge_proximity_logic(grid)

    def _sequence_tail_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute sequence tail scores (private helper)."""
        if not np.issubdtype(grid.dtype, np.integer):
            grid = grid.astype(np.int32)
        rows, cols = grid.shape
        scores = np.zeros((rows, cols), dtype=np.float32)
        blank_mask = (grid == -1)
        
        directions = [(0,1), (1,0), (1,1), (1,-1), (0,-1), (-1,0), (-1,-1), (-1,1)]
        
        for r in nb.prange(rows):
            for c in range(cols):
                if not blank_mask[r, c]:
                    continue
                max_score = 0.0
                for dr, dc in directions:
                    r1, c1 = r + dr, c + dc
                    r2, c2 = r + 2*dr, c + 2*dc
                    if (0 <= r1 < rows and 0 <= c1 < cols and 
                        0 <= r2 < rows and 0 <= c2 < cols and
                        grid[r1, c1] > 0 and grid[r2, c2] > 0):
                        v1, v2 = grid[r1, c1], grid[r2, c2]
                        if (v2 - v1) % 2 == 0:
                            expected = (v1 + v2) // 2
                            if 1 <= expected <= rows * cols:
                                max_score = max(max_score, 0.8)
                scores[r, c] = max_score
        return scores

    @scoring_module
    @nb.njit(parallel=True)
    def sequence_tail_analyzer(self, grid: np.ndarray) -> np.ndarray:
        """Module 2: Sequence tail analysis, arithmetic sequences."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for sequence_tail_analyzer")
            return np.zeros((1, 1), dtype=np.float32)
        return self._sequence_tail_logic(grid)

    @scoring_module
    def connectivity_heatmap(self, grid: np.ndarray) -> np.ndarray:
        """Module 3: Connectivity heatmap with dynamic scaling."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for connectivity_heatmap")
            return np.zeros((1, 1), dtype=np.float32)
        return self._connectivity_heatmap_logic(grid)

    def _entropy_risk_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute entropy risk scores (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        
        scores = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                if not legal_mask[r, c]:
                    continue
                window = grid[max(0, r-1):r+2, max(0, c-1):c+2]
                valid_vals = window[window > 0]
                if len(valid_vals) > 1:
                    counts = np.bincount(valid_vals.astype(np.int32))
                    probs = counts[counts > 0] / (len(valid_vals) + 1e-8)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    scores[r, c] = entropy / (np.log2(len(valid_vals) + 1e-10) or 1.0)
        
        return np.where(legal_mask, scores / (np.max(scores + 1e-8) or 1.0), 0).astype(np.float32)

    @scoring_module
    def entropy_risk_fusion(self, grid: np.ndarray) -> np.ndarray:
        """Module 4: Entropy risk fusion, local entropy."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for entropy_risk_fusion")
            return np.zeros((1, 1), dtype=np.float32)
        return self._entropy_risk_logic(grid)

    def _detect_skip_patterns_logic(self, grid: np.ndarray) -> np.ndarray:
        """Detect row/column skip patterns (private helper)."""
        rows, cols = grid.shape
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        blank_mask = (grid == -1)
        
        for axis in range(2):  # 0 for rows, 1 for columns
            if axis == 0:
                data = grid
                size = cols
            else:
                data = grid.T
                size = rows
                
            for i in range(size):
                row = data[i]
                filled_indices = np.where(row > 0)[0]
                if len(filled_indices) < 2:
                    continue
                differences = np.diff(filled_indices)
                common_diff = np.median(differences) if len(differences) > 0 else 1
                for j in range(size):
                    if blank_mask[i, j] if axis == 0 else blank_mask[j, i]:
                        next_expected = filled_indices[-1] + common_diff if filled_indices.size > 0 else j
                        if abs(j - next_expected) <= 1:
                            if axis == 0:
                                heatmap[i, j] = 0.9
                            else:
                                heatmap[j, i] = 0.9
                            
        return heatmap

    @scoring_module
    def detect_skip_patterns(self, grid: np.ndarray) -> np.ndarray:
        """Module 5: Detect row/column skip patterns."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for detect_skip_patterns")
            return np.zeros((1, 1), dtype=np.float32)
        return self._detect_skip_patterns_logic(grid)

    def _compute_focus_score_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute focus score based on local density (private helper)."""
        kernel = np.ones((3, 3), dtype=np.float32)
        density = convolve2d((grid > 0).astype(np.float32), kernel, mode='same', boundary='symm')
        max_density = np.max(density) or 1.0
        return np.where(grid == -1, density / max_density, 0).astype(np.float32)

    @scoring_module
    def compute_focus_score(self, grid: np.ndarray) -> np.ndarray:
        """Module 6: Compute focus score based on local density."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for compute_focus_score")
            return np.zeros((1, 1), dtype=np.float32)
        return self._compute_focus_score_logic(grid)

    def _detect_mirror_sequences_logic(self, grid: np.ndarray) -> np.ndarray:
        """Detect mirror sequences after horizontal/vertical mirroring (private helper)."""
        rows, cols = grid.shape
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        blank_mask = (grid == -1)
        
        # Horizontal mirror
        h_mirrored = grid[:, ::-1]
        for i in range(rows):
            row = h_mirrored[i]
            filled = row[row > 0]
            if len(filled) >= 2:
                sorted_filled = np.sort(filled)
                for j in range(cols):
                    if blank_mask[i, cols-1-j]:
                        expected = sorted_filled[-1] + 1 if sorted_filled[-1] < rows * cols else 0
                        if expected > 0 and expected == sorted_filled[-2] + 2:
                            heatmap[i, cols-1-j] = 0.8
        
        # Vertical mirror
        v_mirrored = grid[::-1, :]
        for j in range(cols):
            col = v_mirrored[:, j]
            filled = col[col > 0]
            if len(filled) >= 2:
                sorted_filled = np.sort(filled)
                for i in range(rows):
                    if blank_mask[rows-1-i, j]:
                        expected = sorted_filled[-1] + 1 if sorted_filled[-1] < rows * cols else 0
                        if expected > 0 and expected == sorted_filled[-2] + 2:
                            heatmap[rows-1-i, j] = 0.8
        
        return heatmap

    @scoring_module
    def detect_mirror_sequences(self, grid: np.ndarray) -> np.ndarray:
        """Module 7: Detect mirror sequences after horizontal/vertical mirroring."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for detect_mirror_sequences")
            return np.zeros((1, 1), dtype=np.float32)
        return self._detect_mirror_sequences_logic(grid)

    def _compute_difference_trend_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute difference trend scores based on adjacent known numbers (private helper)."""
        rows, cols = grid.shape
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        blank_mask = (grid == -1)
        
        for i in range(rows):
            for j in range(cols):
                if blank_mask[i, j]:
                    neighbors = []
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] > 0:
                            neighbors.append(grid[ni, nj])
                    if len(neighbors) >= 2:
                        differences = np.diff(sorted(neighbors))
                        median_diff = np.median(differences)
                        expected = neighbors[0] + median_diff * (len(neighbors) + 1)
                        if 1 <= expected <= rows * cols:
                            heatmap[i, j] = 0.7 / (1 + abs(expected - np.mean(neighbors)))
        
        return heatmap

    @scoring_module
    def compute_difference_trend(self, grid: np.ndarray) -> np.ndarray:
        """Module 8: Compute difference trend scores based on adjacent known numbers."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error(f"[ERROR] Invalid grid input for compute_difference_trend")
            return np.zeros((1, 1), dtype=np.float32)
        return self._compute_difference_trend_logic(grid)
