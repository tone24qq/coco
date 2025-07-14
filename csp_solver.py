import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ortools.sat.python import cp_model

BLANK = -1


def heuristic_csp_sampling(
    grid: List[List[int]],
    target: int,
    nbr_probs: Dict[Tuple[int, int], float],
    samples: int = 2000,
    enforce_rowcol: bool = False,
) -> Dict[Tuple[int, int], float]:
    """Estimate P(target at cell) using heuristic CSP sampling."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    blanks = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == BLANK]
    used = {grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != BLANK}
    domain = [v for v in range(1, rows * cols + 1) if v not in used]

    counts = {pos: 0 for pos in blanks}

    for _ in range(samples):
        assignment: Dict[Tuple[int, int], int] = {}
        avail = domain.copy()
        order = sorted(blanks, key=lambda pos: -nbr_probs.get(pos, 0.0))
        ok = True
        for pos in order:
            r, c = pos
            if target in avail and random.random() < nbr_probs.get(pos, 0.0):
                val = target
            else:
                choices = avail.copy()
                if target in choices:
                    choices.remove(target)
                val = random.choice(choices) if choices else avail[0]

            if val not in avail:
                ok = False
                break
            if enforce_rowcol:
                for (rr, cc), vv in assignment.items():
                    if vv == val and (rr == r or cc == c):
                        ok = False
                        break
                if not ok:
                    break

            assignment[pos] = val
            avail.remove(val)

        if not ok or len(assignment) != len(blanks):
            continue

        for pos, val in assignment.items():
            if val == target:
                counts[pos] += 1

    return {pos: counts[pos] / samples for pos in blanks}


def full_csp_probabilities(
    grid: List[List[int]],
    target: int,
    *,
    samples: int = 1000,
    reference: Optional[List[List[int]]] = None,
) -> Dict[Tuple[int, int], float]:
    """Use CP-SAT to estimate P(target at cell) with optional reference board."""

    arr = np.asarray(grid, dtype=int)
    rows, cols = arr.shape
    blanks = [(int(r), int(c)) for r, c in np.argwhere(arr == BLANK)]
    if not blanks:
        return {}

    domain = [n for n in range(1, rows * cols + 1) if n not in arr[arr != BLANK]]

    model = cp_model.CpModel()
    vars = [
        model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain), f"x{i}")
        for i in range(len(blanks))
    ]
    model.AddAllDifferent(vars)
    bools = []
    for v in vars:
        b = model.NewBoolVar("b")
        model.Add(v == target).OnlyEnforceIf(b)
        model.Add(v != target).OnlyEnforceIf(b.Not())
        bools.append(b)
    model.Add(sum(bools) == 1)

    if reference is not None:
        ref = np.asarray(reference, dtype=int)
        for (r, c), var in zip(blanks, vars):
            model.AddHint(var, int(ref[r, c]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3
    solver.parameters.num_search_workers = 8

    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables: List[cp_model.IntVar], limit: int) -> None:
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.vars = variables
            self.limit = limit
            self.counts = [0] * len(variables)
            self.solutions = 0

        def on_solution_callback(self) -> None:
            for i, var in enumerate(self.vars):
                if self.Value(var) == target:
                    self.counts[i] += 1
            self.solutions += 1
            if self.solutions >= self.limit:
                self.StopSearch()

    cb = Collector(vars, samples)
    solver.SearchForAllSolutions(model, cb)

    if cb.solutions == 0:
        return {pos: 0.0 for pos in blanks}

    return {blanks[i]: cb.counts[i] / cb.solutions for i in range(len(blanks))}


def csp_with_hint(
    grid: List[List[int]],
    target: int,
    *,
    max_solutions: int = 300,
    time_limit: float = 0.6,
    reference: Optional[List[List[int]]] = None,
    early_stop_eps: float = 0.05,
) -> Dict[Tuple[int, int], float]:
    """Enumerate CP-SAT solutions with AddHint and early stop."""

    arr = np.asarray(grid, dtype=int)
    rows, cols = arr.shape
    blanks = [(int(r), int(c)) for r, c in np.argwhere(arr == BLANK)]
    if not blanks:
        return {}

    domain = [n for n in range(1, rows * cols + 1) if n not in arr[arr != BLANK]]
    model = cp_model.CpModel()
    vars = [
        model.NewIntVarFromDomain(cp_model.Domain.FromValues(domain), f"v{i}")
        for i in range(len(blanks))
    ]
    model.AddAllDifferent(vars)
    bools = []
    for v in vars:
        b = model.NewBoolVar("b")
        model.Add(v == target).OnlyEnforceIf(b)
        model.Add(v != target).OnlyEnforceIf(b.Not())
        bools.append(b)
    model.Add(sum(bools) == 1)

    if reference is not None:
        ref = np.asarray(reference, dtype=int)
        for (r, c), var in zip(blanks, vars):
            model.AddHint(var, int(ref[r, c]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables: List[cp_model.IntVar], limit: int) -> None:
            super().__init__()
            self.vars = variables
            self.limit = limit
            self.counts = [0] * len(variables)
            self.solutions = 0

        def on_solution_callback(self) -> None:
            for i, var in enumerate(self.vars):
                if self.Value(var) == target:
                    self.counts[i] += 1
            self.solutions += 1
            if self.solutions >= self.limit:
                self.StopSearch()
            elif self.solutions >= 50:
                best = max(self.counts)
                second = (
                    sorted(self.counts, reverse=True)[1] if len(self.counts) > 1 else 0
                )
                if (
                    self.solutions > 0
                    and (best - second) / self.solutions > early_stop_eps
                ):
                    self.StopSearch()

    cb = Collector(vars, max_solutions)
    solver.SearchForAllSolutions(model, cb)

    if cb.solutions == 0:
        return {pos: 0.0 for pos in blanks}

    return {blanks[i]: cb.counts[i] / cb.solutions for i in range(len(blanks))}
