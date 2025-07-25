"""CLI demo for CSP solver agent."""

import json

import numpy as np

from agents.csp_solver_agent import predict, solve


def main() -> None:
    board = np.array(
        [
            [1, -1, -1, 4],
            [-1, 4, 1, -1],
            [-1, 1, 4, -1],
            [4, -1, -1, 1],
        ]
    )
    target = 3
    solution = solve(board)
    print("Solved board:\n", solution)
    preds = predict(board, target)
    print("Predictions for", target)
    print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
