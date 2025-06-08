#!/usr/bin/env python3
"""
Main script for scratch card analysis based on SOP and enhanced algorithms.

This script processes scratch card grids, analyzes potential hidden numbers,
and provides predictions with confidence scores. It supports mobile, GitHub,
and service integration with a maximum grid size of 20x20.

Usage:
    python3 script.py <input_file> [--target-num <number>]

Returns:
    Analysis results including best position and confidence scores.
"""
import argparse
import logging
import numpy as np
from analyzer import analyze_board

# 設置日誌
logging.basicConfig(
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed arguments including input file and target number.
    """
    parser = argparse.ArgumentParser(description="Scratch Card Analysis Tool")
    parser.add_argument('input_file', type=str, help='Input file path (JSON/CSV/Excel)')
    parser.add_argument('--target-num', type=int, default=None, help='Target number to predict')
    return parser.parse_args()

def load_grid_from_file(path: str) -> np.ndarray:
    """
    Load grid data from a file.

    Args:
        path (str): Path to the input file.

    Returns:
        np.ndarray: Loaded grid with -1 for unopened cells.

    Raises:
        ValueError: If file format is unsupported or grid exceeds 20x20.
    """
    ext = path.lower().split('.')[-1]
    if ext in ['json']:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        grid = np.array(data)
    elif ext in ['csv']:
        grid = np.genfromtxt(path, delimiter=',', dtype=int, filling_values=-1)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(path, header=None, dtype=str)
        grid = np.array([[int(x) if x.isdigit() else -1 for x in row] for row in df.values])
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Validate grid size (max 20x20)
    if grid.shape[0] > 20 or grid.shape[1] > 20:
        raise ValueError("Grid size exceeds maximum limit of 20x20")
    return grid

def main():
    """
    Main function to execute the scratch card analysis.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    logger.info(f"Processing input file: {args.input_file}")

    try:
        grid = load_grid_from_file(args.input_file)
        score, pred, best_pos = analyze_board(grid, weights={}, return_predictions=True, target_num=args.target_num)
        
        logger.info("Grid Analysis Completed:")
        logger.info(f"Best Position: {best_pos[0][:2] if best_pos else 'None'}")
        if best_pos:
            for pos in best_pos[:3]:
                logger.info(f"Position {pos[:2]}: Score {pos[2]:.2f}, Contributions {pos[3]}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()