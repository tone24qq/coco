import os
import logging
import sys
import argparse
import numpy as np
from typing import List, Dict, Any
from analyzer import predict_scratch_card

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for grid input and iterations."""
    parser = argparse.ArgumentParser(description="Predict hidden numbers in a scratch card grid.")
    parser.add_argument("--grid", type=str, required=True, help="2D grid as a comma-separated string, e.g., '1,2,-1;3,-1,5;-1,4,6'")
    parser.add_argument("--iterations", type=int, default=None, help="Number of Monte Carlo iterations")
    return parser.parse_args()

def parse_grid(grid_str: str) -> List[List[int]]:
    """Parse string input into 2D grid."""
    try:
        rows = grid_str.strip().split(';')
        grid = [[int(x) for x in row.split(',')] for row in rows]
        if not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("All rows must have the same length")
        return grid
    except ValueError as e:
        logging.error(f"Invalid grid format: {e}")
        raise

def main():
    """Main function to run scratch card prediction."""
    args = parse_args()
    try:
        grid = parse_grid(args.grid)
        iterations = args.iterations or (int(os.getenv("ITER", 10_000)) if os.getenv("USE_FORMULA_ONLY") != "1" else 5_000)  # Dynamic iteration
        grid_np = np.array(grid, dtype=np.int64)
        
        # Validate grid
        known_vals = grid_np[grid_np != -1]
        if len(known_vals) != len(np.unique(known_vals)):
            raise ValueError("Grid contains duplicate numbers")
        if any(v < 1 or v > grid_np.size for v in known_vals):
            raise ValueError("Numbers must be between 1 and N")
        
        result = predict_scratch_card(grid, iterations)
        logging.info("Prediction results:")
        for pred in result["predictions"]:
            logging.info(f"Cell ({pred['row']}, {pred['col']}): {pred['candidates']} with confidences {pred['confidences']}")
        logging.info("Full probabilities available in result['full_probabilities']")
        return result
    except (ValueError, Exception) as e:
        logging.error(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()