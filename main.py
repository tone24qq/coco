import os
import logging
import sys
import argparse
import numpy as np
from typing import List, Dict, Any
import psutil
from analyzer import predict_scratch_card

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for grid input and iterations."""
    parser = argparse.ArgumentParser(description="Predict hidden numbers in a scratch card grid.")
    parser.add_argument("--grid", type=str, required=True, help="2D grid as a comma-separated string, e.g., '1,2,-1;3,-1,5;-1,4,6'")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of Monte Carlo iterations")
    parser.add_argument("--target", type=int, default=None, help="Target number to predict")
    return parser.parse_args()

def parse_grid(grid_str: str) -> List[List[int]]:
    """Parse string input into 2D grid."""
    try:
        rows = grid_str.strip().split(';')
        grid = [[int(x) for x in row.split(',')] for row in rows]
        if not all(len(row) == len(grid[0]) for row in grid) or len(grid) < 4 or len(grid) > 20 or len(grid[0]) < 4 or len(grid[0]) > 20:
            raise ValueError("Grid must be 4x4 to 20x20 with consistent row length")
        return grid
    except ValueError as e:
        logger.error(f"Invalid grid format: {e}")
        raise

def main():
    """Main function to run scratch card prediction with resource monitoring."""
    args = parse_args()
    try:
        # Resource monitoring
        if psutil.virtual_memory().percent > 75 or psutil.cpu_percent() > 90:
            logger.warning("High resource usage detected, reducing iterations")
            args.iterations = max(100, args.iterations // 2)

        grid = parse_grid(args.grid)
        iterations = args.iterations
        grid_np = np.array(grid, dtype=np.int16)  # Use int16 for memory efficiency
        
        # Validate grid
        known_vals = grid_np[grid_np != -1]
        rows, cols = grid_np.shape
        max_val = rows * cols
        if len(known_vals) != len(np.unique(known_vals)):
            raise ValueError("Grid contains duplicate numbers")
        if any(v < 1 or v > max_val for v in known_vals):
            raise ValueError(f"Numbers must be between 1 and {max_val}")
        
        logger.info(f"Starting prediction for {rows}x{cols} grid with {iterations} iterations")
        result = predict_scratch_card(grid, target_num=args.target, iterations=iterations)
        
        # Log results with detailed metrics
        logger.info("Prediction results:")
        for pred in result["predictions"]:
            logger.info(f"Cell ({pred['row']}, {pred['col']}): {pred['candidates']} with probability {pred['probability']:.2f}%")
        logger.info(f"Full probabilities available in result['full_probabilities']")
        logger.info(f"Memory usage: {psutil.virtual_memory().percent:.1f}%, CPU usage: {psutil.cpu_percent():.1f}%")
        
        return result
    except (ValueError, Exception) as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()