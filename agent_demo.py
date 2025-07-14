"""Simple CLI to run the matrix factorization agent on a sample board."""

import argparse
import json
from pathlib import Path

import numpy as np

from coco_agents.predict_agent import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction agent")
    parser.add_argument(
        "board", type=Path, help="Path to JSON file containing board list"
    )
    parser.add_argument("target", type=int, help="Target number to search")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    with args.board.open() as f:
        data = json.load(f)
    board = np.array(data)
    preds = predict(board, args.target, rank=args.rank, max_iter=args.max_iter)
    for p in preds[:5]:
        print(f"row={p['row']} col={p['col']} score={p['score']:.3f}")


if __name__ == "__main__":
    main()
