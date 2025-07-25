"""CLI demo for CSP solver agent."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from agents.csp_solver_agent import predict, solve


def _load(path: pathlib.Path) -> np.ndarray:
    data = [list(map(int, line.split())) for line in path.read_text().splitlines()]
    return np.array(data, dtype=int)


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("board", type=pathlib.Path)
    pa.add_argument("-t", "--target", type=int, required=True)
    args = pa.parse_args(argv)

    board = _load(args.board)
    solution = solve(board)
    print("Solved board:\n", solution)
    preds = predict(board, args.target)
    print("Predictions for", args.target)
    print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
