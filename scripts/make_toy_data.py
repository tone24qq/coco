"""Generate synthetic permutation boards for training."""

import argparse
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    N = args.rows * args.cols
    boards = []
    for _ in range(args.num):
        perm = np.random.permutation(np.arange(1, N + 1)).tolist()
        board = [perm[i : i + args.cols] for i in range(0, N, args.cols)]
        boards.append({"board": board})

    with open(args.out, "w", encoding="utf8") as f:
        for b in boards:
            f.write(json.dumps(b) + "\n")


if __name__ == "__main__":
    main()
