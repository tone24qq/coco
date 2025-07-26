"""Demo script to run the MET agent on a sample board."""

import argparse
import json

import numpy as np
import torch

from agents.met_agent import predict
from model import DynamicMET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("board_file", help="Path to JSON file containing a board")
    parser.add_argument("--model", default="met_8x10.pth")
    args = parser.parse_args()

    with open(args.board_file) as f:
        board = np.array(json.load(f)["board"], dtype=int)

    rows, cols = board.shape
    model = DynamicMET(rows * cols, 80)
    ckpt = torch.load(args.model, map_location="cpu")
    model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]

    results = predict(board.copy(), target=int(board[0, 0]), model=model)
    print(results)


if __name__ == "__main__":
    main()
