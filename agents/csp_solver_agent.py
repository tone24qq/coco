"""CSP-based Sudoku solver agent.

This module provides a CSP solver using AC-3, MRV, Forward Checking,
and Backtracking to solve Sudoku-like puzzles.

The ``predict`` function conforms to the interface described in AGENTS.md.
It returns positions likely containing the target number after solving the
puzzle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from coco_common.csp_utils import get_subgrid_indices


@dataclass(frozen=True)
class Cell:
    row: int
    col: int


class ChangeStack(list):
    """Simple stack recording domain changes for fast undo."""

    def snapshot(self) -> int:
        return len(self)

    def push(self, cell: Tuple[int, int], value: int) -> None:
        self.append((cell, value))

    def undo(self, domains: Dict[Tuple[int, int], set], snap: int) -> None:
        while len(self) > snap:
            cell, value = self.pop()
            domains[cell].add(value)


def _get_subgrid_size(n: int) -> int:
    size = int(np.sqrt(n))
    if size * size != n:
        raise ValueError(f"Board size {n} is not a perfect square")
    return size


def _peers(n: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    sub = _get_subgrid_size(n)
    peers: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r in range(n):
        for c in range(n):
            peer_set = set()
            # row and column
            peer_set.update(((r, j) for j in range(n) if j != c))
            peer_set.update(((i, c) for i in range(n) if i != r))
            # subgrid
            sg_r, sg_c = get_subgrid_indices(r, c, sub)
            for i in range(sg_r, sg_r + sub):
                for j in range(sg_c, sg_c + sub):
                    if i == r and j == c:
                        continue
                    peer_set.add((i, j))
            peers[(r, c)] = list(peer_set)
    return peers


def _ac3(
    domains: Dict[Tuple[int, int], set],
    peers: Dict[Tuple[int, int], List[Tuple[int, int]]],
    stack: ChangeStack,
) -> bool:
    queue = [(xi, xj) for xi in domains for xj in peers[xi]]
    while queue:
        xi, xj = queue.pop(0)
        if _revise(domains, xi, xj, stack):
            if not domains[xi]:
                return False
            for xk in peers[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True


def _revise(
    domains: Dict[Tuple[int, int], set],
    xi: Tuple[int, int],
    xj: Tuple[int, int],
    stack: ChangeStack,
) -> bool:
    revised = False
    for x in set(domains[xi]):
        if all(x == y for y in domains[xj]):
            domains[xi].remove(x)
            stack.push(xi, x)
            revised = True
    return revised


def _select_unassigned_variable(
    domains: Dict[Tuple[int, int], set],
) -> Optional[Tuple[int, int]]:
    candidates = [
        (len(values), cell) for cell, values in domains.items() if len(values) > 1
    ]
    return min(candidates, default=(None, None))[1] if candidates else None


def _forward_check(
    domains: Dict[Tuple[int, int], set],
    peers: Dict[Tuple[int, int], List[Tuple[int, int]]],
    cell: Tuple[int, int],
    value: int,
    stack: ChangeStack,
) -> None:
    for peer in peers[cell]:
        if value in domains[peer]:
            domains[peer].remove(value)
            stack.push(peer, value)


def _backtrack(
    domains: Dict[Tuple[int, int], set],
    peers: Dict[Tuple[int, int], List[Tuple[int, int]]],
    stack: ChangeStack,
) -> Optional[Dict[Tuple[int, int], int]]:
    if all(len(v) == 1 for v in domains.values()):
        return {cell: next(iter(values)) for cell, values in domains.items()}

    cell = _select_unassigned_variable(domains)
    if cell is None:
        return None

    values = list(domains[cell])
    for value in values:
        snap = stack.snapshot()
        for val in values:
            if val != value:
                domains[cell].remove(val)
                stack.push(cell, val)
        _forward_check(domains, peers, cell, value, stack)
        if _ac3(domains, peers, stack):
            result = _backtrack(domains, peers, stack)
            if result:
                return result
        stack.undo(domains, snap)
    return None


def solve(board: np.ndarray) -> Optional[np.ndarray]:
    """Solve the given Sudoku puzzle.

    Parameters
    ----------
    board: np.ndarray
        ``n x n`` array with ``-1`` for blanks.

    Returns
    -------
    np.ndarray or None
        Solved board or ``None`` if no solution exists.
    """
    n = board.shape[0]
    if board.shape[0] != board.shape[1]:
        raise ValueError("Board must be square")

    digits = set(range(1, n + 1))
    peers = _peers(n)
    domains: Dict[Tuple[int, int], set] = {}
    for r in range(n):
        for c in range(n):
            value = board[r, c]
            if value == -1:
                domains[(r, c)] = digits.copy()
            else:
                domains[(r, c)] = {int(value)}

    if not _ac3(domains, peers, ChangeStack()):
        return None
    result = _backtrack(domains, peers, ChangeStack())
    if result is None:
        return None
    solved = np.full_like(board, -1)
    for (r, c), val in result.items():
        solved[r, c] = val
    return solved


def predict(board: np.ndarray, target: int, **kwargs: Any) -> List[Dict[str, Any]]:
    """Predict positions of ``target`` using CSP solving.

    Returns a list of dictionaries ``{"row": int, "col": int, "score": float}``.
    Score is ``1.0`` for deterministic predictions.
    """
    solved = solve(board.copy())
    predictions: List[Dict[str, Any]] = []
    if solved is None:
        return predictions
    rows, cols = np.where(solved == target)
    for r, c in zip(rows, cols):
        predictions.append({"row": int(r), "col": int(c), "score": 1.0})
    return predictions
