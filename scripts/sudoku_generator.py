"""Generate random Sudoku puzzles and solutions."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def _pattern(r: int, c: int, sub: int, n: int) -> int:
    return (sub * (r % sub) + r // sub + c) % n


def _shuffle(seq: list[int], rng: np.random.Generator) -> list[int]:
    arr = np.array(seq)
    rng.shuffle(arr)
    return arr.tolist()


def generate(
    n: int, blanks: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if n not in (4, 9):
        raise ValueError("only 4x4 or 9x9 supported")
    rng = rng or np.random.default_rng()
    sub = int(n**0.5)
    rows = [
        g * sub + r
        for g in _shuffle(list(range(sub)), rng)
        for r in _shuffle(list(range(sub)), rng)
    ]
    cols = [
        g * sub + c
        for g in _shuffle(list(range(sub)), rng)
        for c in _shuffle(list(range(sub)), rng)
    ]
    nums = _shuffle(list(range(1, n + 1)), rng)
    board = np.array([[nums[_pattern(r, c, sub, n)] for c in cols] for r in rows])
    puzzle = board.copy()
    blanks = min(blanks, n * n)
    idx = rng.choice(n * n, blanks, replace=False)
    puzzle[np.unravel_index(idx, puzzle.shape)] = -1
    return puzzle, board


def main(argv: list[str] | None = None) -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("-n", type=int, choices=[4, 9], default=9)
    pa.add_argument("-b", "--blanks", type=int, default=None)
    pa.add_argument("-o", "--output", type=pathlib.Path)
    args = pa.parse_args(argv)

    blanks = args.blanks if args.blanks is not None else args.n * args.n // 3
    puzzle, solution = generate(args.n, blanks)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "board.txt").write_text(
            "\n".join(" ".join(map(str, row)) for row in puzzle)
        )
        (args.output / "solution.txt").write_text(
            "\n".join(" ".join(map(str, row)) for row in solution)
        )
    else:
        print("Puzzle:\n", puzzle)
        print("Solution:\n", solution)


if __name__ == "__main__":
    main()
