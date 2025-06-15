# main.py

#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import numpy as np
from typing import List, Dict, Any
from analyzer import predict_scratch_card

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

USE_FORMULA_ONLY = os.getenv("USE_FORMULA_ONLY", "0") == "1"
ITER_ENV = os.getenv("ITER")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict hidden numbers in a scratch card grid.")
    p.add_argument(
        "--grid", "-g", type=str, required=True,
        help="Grid as '1,2,-1;3,-1,5;...'"
    )
    p.add_argument(
        "--iterations", "-n", type=int, default=None,
        help="Override number of Monte Carlo iterations"
    )
    return p.parse_args()

def parse_grid(s: str) -> List[List[int]]:
    rows = [r.strip() for r in s.split(";")]
    grid = [[int(x) for x in r.split(",")] for r in rows]
    if any(len(r) != len(grid[0]) for r in grid):
        logging.error("Each row must have the same length")
        sys.exit(1)
    return grid

def determine_iterations(grid: List[List[int]], cli: int=None) -> int:
    if cli:
        return cli
    if ITER_ENV and ITER_ENV.isdigit():
        return int(ITER_ENV)
    r, c = len(grid), len(grid[0])
    return 10_000_000 if r*c<50 else 5_000_000 if r*c<200 else 1_000_000

def main() -> int:
    args = parse_args()
    grid = parse_grid(args.grid)
    iters = determine_iterations(grid, args.iterations)
    logging.info(f"START predict (formula_only={USE_FORMULA_ONLY}, iter={iters})")
    try:
        out = predict_scratch_card(grid, iters, USE_FORMULA_ONLY)
        for p in out["predictions"]:
            rc = f"({p['row']},{p['col']})"
            nums = p["candidates"]
            conf = [f"{v:.3f}" for v in p["confidences"]]
            logging.info(f"{rc}: {nums} conf={conf}")
        return 0
    except Exception as e:
        logging.error(f"ERROR: {e}")
        return 1

if __name__=="__main__":
    sys.exit(main())