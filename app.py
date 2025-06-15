import logging
from analyzer import predict_scratch_card

def run_prediction(grid_str: str, iterations: int = 5000000, use_formula_only: bool = False):
    import numpy as np

    rows = grid_str.strip().split(";")
    grid = [[int(cell) for cell in row.split(",")] for row in rows]
    grid_np = np.array(grid, dtype=np.int64)
    known_vals = grid_np[grid_np != -1]

    if len(known_vals) != len(np.unique(known_vals)):
        raise ValueError("Duplicate values detected in known cells.")
    if any(v < 1 or v > grid_np.size for v in known_vals):
        raise ValueError(f"Cell values must be in range 1 to {grid_np.size}.")

    result = predict_scratch_card(grid, iterations, use_formula_only=use_formula_only)

    print("=== App Prediction Output ===")
    for pred in result["predictions"]:
        print(f"({pred['row']},{pred['col']}): {pred['candidates']} -> {pred['confidences']}")
    return result

if __name__ == "__main__":
    import os
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="App wrapper for scratch card predictor")
    parser.add_argument("--grid", type=str, required=True, help="Grid string e.g. '1,2,-1;4,5,-1;7,8,9'")
    parser.add_argument("--iterations", type=int, default=None, help="Override ITER value")
    parser.add_argument("--formula-only", action="store_true", help="Force formula-only mode")
    args = parser.parse_args()

    iter_count = args.iterations or int(os.environ.get("ITER", "5000000"))
    formula_flag = args.formula_only or (os.environ.get("USE_FORMULA_ONLY", "0") == "1")

    try:
        run_prediction(args.grid, iterations=iter_count, use_formula_only=formula_flag)
    except Exception as e:
        logging.error(f"App execution failed: {e}")
        sys.exit(1)