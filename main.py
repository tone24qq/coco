import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray

# fmt: off
# isort: off
import analyzer
from analyzer import (
    probability_heatmap,
    predict_scratch_card,
    render_heatmap,
)
# isort: on
# fmt: on

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

priors: Dict[int, float] = {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for grid input and iterations."""
    parser = argparse.ArgumentParser(
        description="Predict hidden numbers in a scratch card grid."
    )
    parser.add_argument(
        "--grid",
        type=str,
        required=True,
        help="2D grid as a comma-separated string, e.g., '1,2,-1;3,-1,5;-1,4,6'",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.getenv("ITER", "5000")),
        help="Number of Monte Carlo iterations",
    )
    parser.add_argument(
        "--global-iter",
        type=int,
        default=None,
        help="Phase-1 global iteration count",
    )
    parser.add_argument(
        "--focus-iter",
        type=int,
        default=None,
        help="Phase-2 focused iteration count",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top cells to refine")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Exploration rate")
    parser.add_argument(
        "--target", type=int, default=None, help="Target number to predict"
    )
    parser.add_argument(
        "--heatmap-k",
        type=int,
        default=None,
        help="Generate probability heatmap for this number (None to skip)",
    )
    parser.add_argument(
        "--heatmap-iter",
        type=int,
        default=1000,
        help="Iterations for heatmap simulation",
    )
    parser.add_argument(
        "--heatmap-format",
        type=str,
        choices=["raw", "base64", "png_bytes"],
        default="png_bytes",
        help="Format for heatmap output",
    )
    parser.add_argument(
        "--sample-gamma",
        type=float,
        default=0.0,
        help="Weight for sample-based frequency prior",
    )
    return parser.parse_args()


def parse_grid(grid_str: str) -> List[List[int]]:
    """Parse string input into 2D grid."""
    try:
        rows = grid_str.strip().split(";")
        grid = [[int(x) for x in row.split(",")] for row in rows]
        grid_np = np.array(grid, dtype=int)
        if grid_np.ndim != 2:
            raise ValueError("Grid must be a 2D matrix")
        r, c = grid_np.shape
        if r < 2 or c < 2:
            raise ValueError("Grid must be at least 2x2 with consistent row length")
        return grid_np.tolist()
    except ValueError as e:
        logging.error(f"Invalid grid format: {e}")
        raise


def main():
    """Main function to run scratch card prediction."""
    args = parse_args()
    try:
        p = Path("output/cleaned_data.json")
        global priors
        if p.exists():
            priors = json.loads(p.read_text(encoding="utf-8"))
        else:
            priors = {}
        grid = parse_grid(args.grid)
        iterations = args.iterations
        grid_np = np.array(grid, dtype=np.int64)

        # Validate grid
        known_vals = grid_np[grid_np != -1]
        rows, cols = grid_np.shape
        max_val = rows * cols
        if known_vals.size != np.unique(known_vals).size:
            raise ValueError("Grid contains duplicate numbers")
        if np.any((known_vals < 1) | (known_vals > max_val)):
            raise ValueError(f"Numbers must be between 1 and {max_val}")

        # Disable Ray dashboard to avoid excessive port scanning
        ray.init(num_cpus=4, include_dashboard=False)
        result = predict_scratch_card(
            grid,
            target_num=args.target,
            iterations=iterations,
            global_iter=args.global_iter,
            focus_iter=args.focus_iter,
            top_n=args.top_n,
            epsilon=args.epsilon,
            result_top_k=None,
            priors=priors,
            sample_gamma=args.sample_gamma,
        )
        ray.shutdown()

        if args.heatmap_k is not None:
            prob = probability_heatmap(
                grid_np,
                args.heatmap_k if args.heatmap_k != -1 else None,
                args.heatmap_iter,
            )
            if isinstance(prob, dict):
                logging.info("Full probability maps computed (no image)")
            else:
                rendered = render_heatmap(prob, args.heatmap_format)
                if isinstance(rendered, bytes):
                    with open("heatmap.png", "wb") as f:
                        f.write(rendered)
                    logging.info("Heatmap saved to heatmap.png")
                elif isinstance(rendered, str):
                    with open("heatmap.txt", "w") as f:
                        f.write(rendered)
                    logging.info("Heatmap base64 saved to heatmap.txt")
        logging.info("Prediction results:")
        for pred in result["predictions"]:
            r = int(pred["row"]) + 1
            c = int(pred["col"]) + 1
            logging.info(
                f"Cell ({r}, {c}): {pred['candidates']} with probability {pred['probability']:.2f}%"
            )
        logging.info("Full probabilities available in result['full_probabilities']")
        logging.info("Complete!")
        return result
    except (ValueError, Exception) as e:
        logging.error(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "dump_prior":
        analyzer.dump_prior(sys.argv[2], sys.argv[3])
    else:
        main()
