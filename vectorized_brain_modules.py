"""
vectorized_brain_modules.py - Vectorized implementation of 4 modules with dynamic heatmap scaling
"""
import numpy as np
from scipy.ndimage import convolve, zoom
import numba as nb
import logging
import time
import json
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def scoring_module(fn):
    """Decorator to mark a method as a scoring module."""
    fn.is_scoring = True
    return fn

class VectorizedBrainModules:
    """Vectorized analyzer with 4 modules"""
    
    def __init__(self):
        """Initialize with cached heatmap from JSON samples"""
        self.heatmap_cache = None
        self._load_heatmap()
        self.module_list = [method for method in self.__class__.__dict__.values() 
                           if callable(method) and getattr(method, 'is_scoring', False)]

    def _load_heatmap(self) -> None:
        """Load JSON samples and build normalized heatmap"""
        try:
            samples_dir = Path(__file__).parent / "samples" / "data"
            if not samples_dir.exists():
                logger.warning(f"Samples directory {samples_dir} does not exist")
                self.heatmap_cache = None
                return
            
            json_files = list(samples_dir.glob("*.json"))
            if not json_files:
                logger.warning(f"No JSON files found in {samples_dir}")
                self.heatmap_cache = None
                return
            
            first = json.loads(json_files[0].read_text(encoding='utf-8'))
            rows, cols = len(first["grid"]), len(first["grid"][0])
            heatmap = np.zeros((rows, cols), dtype=np.int32)
            
            for json_file in json_files:
                try:
                    data = json.loads(json_file.read_text(encoding='utf-8'))
                    r = data["answer"]["row"] - 1  # 1-based to 0-based
                    c = data["answer"]["col"] - 1
                    if 0 <= r < rows and 0 <= c < cols:
                        heatmap[r, c] += 1
                    else:
                        logger.warning(f"Invalid position in {json_file}: row={r+1}, col={c+1}")
                except Exception as e:
                    logger.error(f"Failed to load {json_file}: {e}")
            
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val > min_val:
                self.heatmap_cache = ((heatmap - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
            else:
                self.heatmap_cache = np.zeros_like(heatmap, dtype=np.float32)
            logger.info(f"Loaded heatmap from {len(json_files)} JSON samples, shape={heatmap.shape}")
        except Exception as e:
            logger.error(f"Heatmap loading failed: {e}")
            self.heatmap_cache = None

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
        kernel = 1.0 / (distances ** 1.5)
        kernel[radius, radius] = 0
        
        value_weights = np.where(filled_mask, grid * 0.1, 0)
        scores = convolve(value_weights, kernel, mode='constant')
        
        max_val = np.max(scores[legal_mask]) if np.any(legal_mask) else 1.0
        return np.where(legal_mask, scores / max_val, 0)

    @scoring_module
    def edge_proximity_fusion(self, grid: np.ndarray) -> np.ndarray:
        """Module 1: Edge proximity fusion, low values near edges"""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for edge_proximity_fusion")
            return np.zeros_like(grid, dtype=np.float32)
        return self._edge_proximity_logic(grid)

    def _sequence_tail_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute sequence tail scores (private helper)."""
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
        """Module 2: Sequence tail analysis, arithmetic sequences"""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for sequence_tail_analyzer")
            return np.zeros_like(grid, dtype=np.float32)
        return self._sequence_tail_logic(grid)

    def _connectivity_heatmap_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute connectivity heatmap scores (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        
        if self.heatmap_cache is None:
            logger.warning("No valid heatmap, using uniform scores")
            return np.where(legal_mask, 0.5, 0)
        else:
            zoom_factors = (rows / self.heatmap_cache.shape[0], cols / self.heatmap_cache.shape[1])
            resized_heatmap = zoom(self.heatmap_cache, zoom_factors, order=1)
            scores = np.where(legal_mask, resized_heatmap, 0)
            return scores / np.max(scores + 1e-8)

    @scoring_module
    def connectivity_heatmap(self, grid: np.ndarray) -> np.ndarray:
        """Module 3: Connectivity heatmap with dynamic scaling"""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for connectivity_heatmap")
            return np.zeros_like(grid, dtype=np.float32)
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
                    counts = np.bincount(valid_vals.astype(int))
                    probs = counts[counts > 0] / len(valid_vals)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    scores[r, c] = entropy / np.log2(len(valid_vals) + 1e-10)
        
        return np.where(legal_mask, scores / np.max(scores + 1e-8), 0)

    @scoring_module
    def entropy_risk_fusion(self, grid: np.ndarray) -> np.ndarray:
        """Module 4: Entropy risk fusion, local entropy"""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for entropy_risk_fusion")
            return np.zeros_like(grid, dtype=np.float32)
        return self._entropy_risk_logic(grid)

    class VectorizedBrainModules:
    ...

    def _detect_skip_patterns_logic(self, grid: np.ndarray) -> np.ndarray:
        """Detect row/column skip patterns (private helper)."""
        rows, cols = grid.shape
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        blank_mask = (grid == -1)
        
        for axis in range(2):
            data = grid if axis == 0 else grid.T
            size = rows if axis == 0 else cols

            for i in range(size):
                row = data[i]
                filled_indices = np.where(row > 0)[0]
                if len(filled_indices) < 2:
                    continue
                differences = np.diff(filled_indices)
                common_diff = np.median(differences) if len(differences) > 0 else 1

                for j in range(rows if axis == 0 else cols):
                    is_blank = blank_mask[i, j] if axis == 0 else blank_mask[j, i]
                    if not is_blank:
                        continue
                    next_expected = filled_indices[-1] + common_diff
                    if abs(j - next_expected) <= 1:
                        if axis == 0:
                            heatmap[i, j] = 0.9
                        else:
                            heatmap[j, i] = 0.9

        return heatmap

    @scoring_module
    def detect_skip_patterns(self, grid: np.ndarray) -> np.ndarray:
        """Detect row/column skip patterns and return a heatmap."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for detect_skip_patterns")
            return np.zeros_like(grid, dtype=np.float32)
        return self._detect_skip_patterns_logic(grid)

    
    def _compute_focus_score_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute focus score based on local density (private helper).
        
        Args:
            grid (np.ndarray): 2D integer array with -1 indicating blank cells.
            
        Returns:
            np.ndarray: 2D heatmap with scores based on 3x3 window density.
            
        Notes:
            Uses convolution with a 3x3 kernel, normalized by max density.
        """
        from scipy.signal import convolve2d
        kernel = np.ones((3, 3), dtype=np.float32)
        density = convolve2d((grid > 0).astype(np.float32), kernel, mode='same', boundary='symm')
        max_density = np.max(density)
        return np.where(grid == -1, density / (max_density + 1e-8), 0)

    @scoring_module
    def compute_focus_score(self, grid: np.ndarray) -> np.ndarray:
        """Compute focus score based on local density of known numbers."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for compute_focus_score")
            return np.zeros_like(grid, dtype=np.float32)
        return self._compute_focus_score_logic(grid)

    def _detect_mirror_sequences_logic(self, grid: np.ndarray) -> np.ndarray:
        """Detect mirror sequences after horizontal/vertical mirroring (private helper).
        
        Args:
            grid (np.ndarray): 2D integer array with -1 indicating blank cells.
            
        Returns:
            np.ndarray: 2D heatmap with scores for mirror sequence completions.
            
        Notes:
            Assigns 0.8 score if mirroring suggests a consecutive number.
        """
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
                        if expected == sorted_filled[-2] + 2:
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
                        if expected == sorted_filled[-2] + 2:
                            heatmap[rows-1-i, j] = 0.8
        
        return heatmap

    @scoring_module
    def detect_mirror_sequences(self, grid: np.ndarray) -> np.ndarray:
        """Detect mirror sequences after horizontal/vertical mirroring."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for detect_mirror_sequences")
            return np.zeros_like(grid, dtype=np.float32)
        return self._detect_mirror_sequences_logic(grid)

    def _compute_difference_trend_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute difference trend scores based on adjacent known numbers (private helper).
        
        Args:
            grid (np.ndarray): 2D integer array with -1 indicating blank cells.
            
        Returns:
            np.ndarray: 2D heatmap with scores based on arithmetic progression likelihood.
            
        Notes:
            Scores are based on median difference of neighbors, normalized by deviation.
        """
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
        """Compute difference trend scores based on adjacent known numbers."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for compute_difference_trend")
            return np.zeros_like(grid, dtype=np.float32)
        return self._compute_difference_trend_logic(grid)

    def test_with_masking(
        self, original_grid: np.ndarray, n_mask: int = 40, target: int = 7, n_trials: int = 20
    ) -> Tuple[float, float]:
        """Random masking test for arbitrary grid sizes"""
        try:
            true_positions = np.where(original_grid == target)
            if len(true_positions[0]) == 0:
                raise ValueError(f"Target number {target} not found in grid")
            
            accuracies = []
            max_mask = min(n_mask, original_grid.size - len(true_positions[0]))
            
            for trial in range(n_trials):
                grid = original_grid.copy()
                valid_indices = np.where(grid != target)
                indices = np.random.choice(
                    np.arange(grid.size)[valid_indices[0] * grid.shape[1] + valid_indices[1]], 
                    max_mask, 
                    replace=False
                )
                grid.flat[indices] = -1
                
                for r, c in zip(true_positions[0], true_positions[1]):
                    grid[r, c] = -1
                
                from analyzer11_optimized import analyze_with_prior
                results = analyze_with_prior(grid, target, request_id=f"test_{trial}")
                
                correct = False
                for r, c, conf in results:
                    if original_grid[r, c] == target:
                        correct = True
                        logger.info(f"Trial {trial+1}: Predicted {(r,c)}, confidence {conf:.3f}, correct")
                        break
                if not correct:
                    logger.info(f"Trial {trial+1}: Prediction failed")
                
                accuracies.append(1.0 if correct else 0.0)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            logger.info(f"Average hit rate: {mean_acc:.3f} ({mean_acc*100:.1f}%), std: {std_acc:.3f}")
            return mean_acc, std_acc
        except Exception as e:
            logger.error(f"Masking test failed: {e}")
            return 0.0, 0.0

def performance_test():
    """Performance test for arbitrary grid"""
    test_grid = np.random.randint(-1, 400, (20, 20))
    test_grid[test_grid == 0] = -1
    brain = VectorizedBrainModules()
    
    start_time = time.time()
    scores = [
        brain.edge_proximity_fusion(test_grid),
        brain.sequence_tail_analyzer(test_grid),
        brain.connectivity_heatmap(test_grid),
        brain.entropy_risk_fusion(test_grid)
    ]
    elapsed = time.time() - start_time
    
    print(f"4 modules computed in {elapsed:.4f} seconds")
    return scores

def run_masking_test():
    """Run random masking test for 20x20 grid"""
    sample_grid = np.arange(1, 401).reshape(20, 20)
    sample_grid[10, 10] = 7
    brain = VectorizedBrainModules()
    np.random.seed(42)
    mean_acc, std_acc = brain.test_with_masking(sample_grid, n_mask=40, target=7, n_trials=20)
    print(f"Final result: Average hit rate {mean_acc*100:.1f}%, std {std_acc:.3f}")