import random
from typing import Dict, List, Tuple

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
