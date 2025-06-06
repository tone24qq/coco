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
logging.basicConfig(level=logging.INFO)

def scoring_module(fn):
    """Decorator to mark a method as a scoring module."""
    fn.is_scoring = True
    return fn

class VectorizedBrainModules:
    """Vectorized analyzer with multiple scoring modules for grid analysis."""
    
    def __init__(self):
        """Initialize with cached heatmap from JSON samples."""
        self.heatmap_cache = {}
        self._load_heatmap()
        self.module_list = [
            method for method in self.__class__.__dict__.values() 
            if callable(method) and getattr(method, 'is_scoring', False)
        ]
        logger.info(f"Initialized VectorizedBrainModules with {len(self.module_list)} scoring modules")

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

            size_to_files = {}
            for json_file in json_files:
                try:
                    data = json.loads(json_file.read_text(encoding='utf-8'))
                    grid = data.get("grid")
                    if not isinstance(grid, list) or not grid:
                        logger.warning(f"Invalid or empty grid in {json_file.name}, skipping")
                        continue
                    rows, cols = len(grid), len(grid[0])
                    if not all(isinstance(row, list) and len(row) == cols for row in grid):
                        logger.warning(f"Inconsistent row lengths in {json_file.name}, skipping")
                        continue
                    size_key = (rows, cols)
                    size_to_files.setdefault(size_key, []).append((json_file, data))
                except Exception as e:
                    logger.warning(f"Failed to parse {json_file.name}: {e}, skipping")

            # Create heatmap for each size
            for size_key, files in size_to_files.items():
                rows, cols = size_key
                heatmap = np.zeros((rows, cols), dtype=np.int32)
                valid_files = 0

                for json_file, data in files:
                    try:
                        ans = data.get("answer")
                        if isinstance(ans, dict) and "row" in ans and "col" in ans:
                            r, c = ans["row"] - 1, ans["col"] - 1  # 1-based to 0-based
                        elif isinstance(ans, list) and len(ans) == 2:
                            r, c = ans[0] - 1, ans[1] - 1  # Support list format
                        else:
                            logger.warning(f"Invalid answer format in {json_file.name}, skipping")
                            continue

                        # Check valid range
                        if not (0 <= r < rows and 0 <= c < cols):
                            logger.warning(
                                f"答案超出 {json_file.name} 的範圍：row={ans[0]}, col={ans[1]}, grid_size=({rows},{cols})"
                            )
                            continue

                        heatmap[r, c] += 1
                        valid_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to process {json_file.name}: {e}, skipping")

                # Normalize heatmap
                if valid_files > 0:
                    min_val, max_val = heatmap.min(), heatmap.max()
                    if max_val > min_val:
                        norm = (heatmap - min_val) / (max_val - min_val + 1e-8)
                    else:
                        norm = np.zeros_like(heatmap, dtype=np.float32)
                    self.heatmap_cache[size_key] = norm.astype(np.float32)
                    logger.info(
                        f"Loaded heatmap for size {size_key} from {valid_files} valid JSON samples"
                    )
                else:
                    # No valid samples for this size
                    logger.warning(f"大小 {size_key} 沒有有效答案，跳過熱圖")
                    # Still create a zeroed entry so caching knows this size was attempted
                    self.heatmap_cache[size_key] = np.zeros((rows, cols), dtype=np.float32)

            if not self.heatmap_cache:
                logger.warning("No valid heatmaps loaded")
        except Exception as e:
            logger.error(f"Heatmap loading failed: {e}")
            self.heatmap_cache = {}

    def _connectivity_heatmap_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute connectivity heatmap scores (private helper).
        
        Args:
            grid (np.ndarray): 2D integer array with -1 indicating blank cells.
            
        Returns:
            np.ndarray: 2D heatmap with scores based on cached heatmap.
        """
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        
        if not self.heatmap_cache:
            logger.warning("No valid heatmap available, using uniform scores")
            return np.where(legal_mask, 0.5, 0).astype(np.float32)
        
        # Find closest heatmap size
        target_size = (rows, cols)
        if target_size in self.heatmap_cache:
            heatmap = self.heatmap_cache[target_size]
            logger.debug(f"Using exact heatmap match for size {target_size}")
        else:
            # Choose closest size by minimizing area difference
            size_diffs = [
                (size, abs(size[0] * size[1] - rows * cols)) 
                for size in self.heatmap_cache
            ]
            closest_size = min(size_diffs, key=lambda x: x[1])[0]
            heatmap = self.heatmap_cache[closest_size]
            logger.debug(f"Using closest heatmap size {closest_size} for grid size {target_size}")
        
        # Resize heatmap to match grid
        zoom_factors = (rows / heatmap.shape[0], cols / heatmap.shape[1])
        resized_heatmap = zoom(heatmap, zoom_factors, order=1)
        scores = np.where(legal_mask, resized_heatmap, 0)
        
        max_score = np.max(scores) + 1e-8
        return (scores / max_score).astype(np.float32)

    @scoring_module
    def connectivity_heatmap(self, grid: np.ndarray) -> np.ndarray:
        """Module 1: Return heatmap-based connectivity scores for blank cells."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for connectivity_heatmap")
            return np.zeros((1, 1), dtype=np.float32)
        return self._connectivity_heatmap_logic(grid)

    def _edge_proximity_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute edge proximity scores (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        
        # Distance to nearest filled cell (Manhattan)
        distance_map = np.full((rows, cols), np.inf, dtype=np.float32)
        filled_positions = np.argwhere(grid > 0)
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == -1:
                    # compute min Manhattan distance to any filled cell
                    dist = np.min(np.abs(filled_positions - np.array([i, j])).sum(axis=1)) \
                           if filled_positions.size else max(rows, cols)
                    distance_map[i, j] = dist

        # Invert distance so smaller actual distance → higher score
        max_dist = np.nanmax(distance_map[np.isfinite(distance_map)]) if np.isfinite(distance_map).any() else 1
        score_map = (max_dist - distance_map) / (max_dist + 1e-8)
        return np.where(legal_mask, score_map.astype(np.float32), 0)

    @scoring_module
    def edge_proximity(self, grid: np.ndarray) -> np.ndarray:
        """Module 2: Score blank cells by proximity to known (non-blank) cells."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for edge_proximity")
            return np.zeros((1, 1), dtype=np.float32)
        return self._edge_proximity_logic(grid)

    def _uniform_score_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute uniform scores for blank cells."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        return np.where(legal_mask, 1.0 / (rows * cols), 0).astype(np.float32)

    @scoring_module
    def uniform_score(self, grid: np.ndarray) -> np.ndarray:
        """Module 3: Return uniform distribution over blank cells."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for uniform_score")
            return np.zeros((1, 1), dtype=np.float32)
        return self._uniform_score_logic(grid)

    def _center_bias_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute center-bias scores (private helper)."""
        rows, cols = grid.shape
        x = np.arange(cols) - (cols - 1) / 2
        y = (np.arange(rows) - (rows - 1) / 2)[:, np.newaxis]
        d2 = x**2 + y**2
        max_d2 = np.max(d2)
        bias = 1 - (d2 / (max_d2 + 1e-8))
        legal_mask = (grid == -1).astype(np.float32)
        return np.where(legal_mask, bias.astype(np.float32), 0)

    @scoring_module
    def center_bias(self, grid: np.ndarray) -> np.ndarray:
        """Module 4: Score blank cells by distance to center of grid."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for center_bias")
            return np.zeros((1, 1), dtype=np.float32)
        return self._center_bias_logic(grid)

    def _neighbor_count_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute neighbor-based scores (private helper)."""
        rows, cols = grid.shape
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0
        filled_mask = (grid > 0).astype(np.float32)
        neighbor_count = convolve2d(filled_mask, kernel, mode="same", boundary="fill", fillvalue=0)
        legal_mask = (grid == -1).astype(np.float32)
        max_count = np.max(neighbor_count) if neighbor_count.size else 1
        scores = neighbor_count / (max_count + 1e-8)
        return np.where(legal_mask, scores.astype(np.float32), 0)

    @scoring_module
    def neighbor_count(self, grid: np.ndarray) -> np.ndarray:
        """Module 5: Score blank cells by count of adjacent filled cells."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for neighbor_count")
            return np.zeros((1, 1), dtype=np.float32)
        return self._neighbor_count_logic(grid)

    def _row_column_balance_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute row-column balance scores (private helper)."""
        rows, cols = grid.shape
        filled = (grid > 0).astype(np.float32)
        row_sum = np.sum(filled, axis=1)[:, np.newaxis]
        col_sum = np.sum(filled, axis=0)[np.newaxis, :]
        total = row_sum + col_sum
        legal_mask = (grid == -1).astype(np.float32)
        max_val = np.max(total) if total.size else 1
        scores = 1 - (total / (max_val + 1e-8))
        return np.where(legal_mask, scores.astype(np.float32), 0)

    @scoring_module
    def row_column_balance(self, grid: np.ndarray) -> np.ndarray:
        """Module 6: Score blank cells by balancing filled counts in row/column."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for row_column_balance")
            return np.zeros((1, 1), dtype=np.float32)
        return self._row_column_balance_logic(grid)

    def _predict_arithmetic_sequence_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute arithmetic sequence prediction scores (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        # Check each blank position for possible arithmetic sequence in row or column
        for i in range(rows):
            row_vals = grid[i, :]
            known = np.argwhere(row_vals > 0).flatten()
            if known.size >= 2:
                diffs = np.diff(row_vals[known])
                if np.all(diffs == diffs[0]):
                    expected = row_vals[known[0]] + diffs[0] * (np.arange(cols) - known[0])
                    for j in np.where(legal_mask[i, :] == 1)[0]:
                        if expected[j] == expected[j]:  # not NaN
                            heatmap[i, j] += 0.8
        for j in range(cols):
            col_vals = grid[:, j]
            known = np.argwhere(col_vals > 0).flatten()
            if known.size >= 2:
                diffs = np.diff(col_vals[known])
                if np.all(diffs == diffs[0]):
                    expected = col_vals[known[0]] + diffs[0] * (np.arange(rows) - known[0])
                    for i in np.where(legal_mask[:, j] == 1)[0]:
                        if expected[i] == expected[i]:
                            heatmap[i, j] += 0.8
        return np.where(legal_mask, heatmap, 0).astype(np.float32)

    @scoring_module
    def predict_arithmetic_sequence(self, grid: np.ndarray) -> np.ndarray:
        """Module 7: Predict blank by arithmetic sequences in rows/columns."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for predict_arithmetic_sequence")
            return np.zeros((1, 1), dtype=np.float32)
        return self._predict_arithmetic_sequence_logic(grid)

    def _compute_difference_trend_logic(self, grid: np.ndarray) -> np.ndarray:
        """Compute difference trend scores based on adjacent known numbers (private helper)."""
        rows, cols = grid.shape
        legal_mask = (grid == -1).astype(np.float32)
        heatmap = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == -1:
                    # gather known neighbors in 8 directions
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] > 0:
                                neighbors.append(grid[ni, nj])
                    if len(neighbors) >= 2:
                        differences = np.diff(sorted(neighbors))
                        median_diff = np.median(differences)
                        expected = neighbors[0] + median_diff * (len(neighbors) + 1)
                        if 1 <= expected <= rows * cols:
                            heatmap[i, j] = 0.7 / (1 + abs(expected - np.mean(neighbors)))
        return np.where(legal_mask, heatmap, 0).astype(np.float32)

    @scoring_module
    def compute_difference_trend(self, grid: np.ndarray) -> np.ndarray:
        """Module 8: Compute difference trend scores based on adjacent known numbers."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            logger.error("Invalid grid input for compute_difference_trend")
            return np.zeros((1, 1), dtype=np.float32)
        return self._compute_difference_trend_logic(grid)


# Example usage and test harness
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and inspect heatmaps from JSON samples.")
    parser.add_argument(
        "--list-sizes",
        action="store_true",
        help="List all grid sizes for which heatmaps have been loaded."
    )
    parser.add_argument(
        "--show-heatmap",
        nargs=2,
        type=int,
        metavar=('ROWS', 'COLS'),
        help="Print the normalized heatmap for a specific size (ROWS, COLS)."
    )
    args = parser.parse_args()

    vbm = VectorizedBrainModules()

    if args.list_sizes:
        sizes = sorted(vbm.heatmap_cache.keys())
        print("Loaded heatmap sizes:")
        for size in sizes:
            print(f"  - {size}, sum of values: {np.sum(vbm.heatmap_cache[size]):.4f}")
        exit(0)

    if args.show_heatmap:
        size = (args.show_heatmap[0], args.show_heatmap[1])
        hm = vbm.heatmap_cache.get(size)
        if hm is None:
            print(f"No heatmap found for size {size}. Available: {list(vbm.heatmap_cache.keys())}")
        else:
            print(f"Heatmap for size {size}:")
            print(hm)
        exit(0)

    # If no args provided, simply list sizes
    sizes = sorted(vbm.heatmap_cache.keys())
    if sizes:
        print("Loaded heatmap sizes:")
        for size in sizes:
            print(f"  - {size}, sum of values: {np.sum(vbm.heatmap_cache[size]):.4f}")
    else:
        print("No valid heatmaps loaded.")
