"""
vectorized_brain_modules.py - Vectorized implementation of 4 modules with JSON-based heatmap
"""
import numpy as np
from scipy.ndimage import convolve
import numba as nb
import logging
import time
import json
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

class VectorizedBrainModules:
    """Vectorized analyzer with 4 modules"""
    
    def __init__(self):
        """Initialize with cached heatmap from JSON samples"""
        self.heatmap_cache = None
        self._load_heatmap()

    def _load_heatmap(self) -> None:
        """Load JSON samples and build normalized heatmap (from 讀取熱力圖教學.txt)"""
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
            
            # Normalize to 0-1 (avoid division by zero)
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val > min_val:
                self.heatmap_cache = ((heatmap - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
            else:
                self.heatmap_cache = np.zeros_like(heatmap, dtype=np.float32)
            logger.info(f"Loaded heatmap from {len(json_files)} JSON samples, shape={heatmap.shape}")
        except Exception as e:
            logger.error(f"Heatmap loading failed: {e}")
            self.heatmap_cache = None

    def edge_proximity_fusion(self, grid: np.ndarray) -> np.ndarray:
        """Module 1: Edge proximity fusion, low values near edges"""
        try:
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
            scores = np.where(legal_mask, scores / max_val, 0)
            return scores
        except Exception as e:
            logger.error(f"EdgeProximityFusion failed: {e}")
            return np.zeros_like(grid, dtype=np.float32)

    @nb.njit(parallel=True)
    def sequence_tail_analyzer(grid: np.ndarray) -> np.ndarray:
        """Module 2: Sequence tail analysis, arithmetic sequences"""
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

    def connectivity_heatmap(self, grid: np.ndarray) -> np.ndarray:
        """Module 3: Connectivity heatmap from JSON samples (from 讀取熱力圖教學.txt)"""
        try:
            rows, cols = grid.shape
            legal_mask = (grid == -1).astype(np.float32)
            
            if self.heatmap_cache is None or self.heatmap_cache.shape != (rows, cols):
                logger.warning("No valid heatmap available, using uniform scores")
                scores = np.where(legal_mask, 0.5, 0)
            else:
                scores = np.where(legal_mask, self.heatmap_cache, 0)
            
            return scores
        except Exception as e:
            logger.error(f"ConnectivityHeatmap failed: {e}")
            return np.zeros_like(grid, dtype=np.float32)

    def entropy_risk_fusion(self, grid: np.ndarray) -> np.ndarray:
        """Module 4: Entropy risk fusion, local entropy"""
        try:
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
            
            scores = np.where(legal_mask, scores / np.max(scores + 1e-8), 0)
            return scores
        except Exception as e:
            logger.error(f"EntropyRiskFusion failed: {e}")
            return np.zeros_like(grid, dtype=np.float32)

    def test_with_masking(
        self, original_grid: np.ndarray, n_mask: int = 40, target: int = 7, n_trials: int = 20
    ) -> Tuple[float, float]:
        """Random masking test to compute hit rate"""
        try:
            true_positions = np.where(original_grid == target)
            if len(true_positions[0]) == 0:
                raise ValueError(f"Target number {target} not found in grid")
            
            accuracies = []
            for trial in range(n_trials):
                grid = original_grid.copy()
                indices = np.random.choice(original_grid.size, n_mask, replace=False)
                grid.flat[indices] = -1
                
                for r, c in zip(true_positions[0], true_positions[1]):
                    grid[r, c] = -1
                
                from analyzer11_optimized import analyze_with_prior
                results = analyze_with_prior(grid, target, request_id=f"test_{trial}")
                
                correct = False
                for r, c, conf in results:
                    if original_grid[r, c] == target:
                        correct = True
                        logger.info(f"Trial {trial+1}: Predicted position {(r,c)}, confidence {conf:.3f}, correct")
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
    """Performance test"""
    test_grid = np.random.randint(-1, 60, (6, 10))
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
    """Run random masking test"""
    sample_grid = np.array([
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        [51, 52, 53, 54, 55, 56,  7, 58, 59, 60]
    ])
    brain = VectorizedBrainModules()
    np.random.seed(42)
    mean_acc, std_acc = brain.test_with_masking(sample_grid, n_mask=40, target=7, n_trials=20)
    print(f"Final result: Average hit rate {mean_acc*100:.1f}%, std {std_acc:.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    performance_test()
    run_masking_test()