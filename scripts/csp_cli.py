"""Command line interface for CSP solver and predictions."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from agents.csp_solver_agent import predict, solve


def _load(path: pathlib.Path) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in path.read_text().splitlines()]
    return np.array(rows, dtype=int)


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("board", type=pathlib.Path)
    pa.add_argument("-t", "--target", type=int, required=True)
    pa.add_argument("--solve", action="store_true")
    args = pa.parse_args(argv)

    board = _load(args.board)
    if args.solve:
        solved = solve(board)
        print(solved if solved is not None else "\u274c no solution")
    else:
        preds = predict(board, args.target)
        print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
