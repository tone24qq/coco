def analyze(board, target):
    """
    Brain3: Diagonal Consecutive Sequence Analysis.
    Checks diagonals (both \ and / directions) for sequences of consecutive numbers with one missing spot.
    If the target fits exactly into a missing spot in an otherwise consecutive diagonal sequence, flag that cell.
    """
    results = []
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0

    # Check main diagonals (down-right direction) for a run of 3 with one missing number
    for r in range(rows - 2):
        for c in range(cols - 2):
            a = board[r][c]           # first in diagonal triple
            b = board[r+1][c+1]       # second (middle) in diagonal
            d = board[r+2][c+2]       # third in diagonal
            # If exactly one of (a, b, d) is None, see if the other two form a consecutive pair around the missing one
            if [a, b, d].count(None) == 1:
                # Case 1: middle is missing
                if b is None and a is not None and d is not None:
                    if abs(d - a) == 2:  # numbers a and d differ by 2 => the middle should be their average
                        mid_val = (a + d) // 2
                        if mid_val == target:
                            results.append({
                                "row": r+1, "col": c+1,
                                "confidence": 0.95,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {a} and {d} on a main diagonal are consecutive around missing {mid_val} (target)."
                            })
                # Case 2: first is missing (a is None)
                if a is None and b is not None and d is not None:
                    if abs(d - b) == 1:
                        # b and d are consecutive; target could be the predecessor or successor to extend the sequence
                        if d - b == 1 and b - 1 == target:
                            results.append({
                                "row": r, "col": c,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {b} and {d} are consecutive; {target} could precede them on this diagonal."
                            })
                        if b - d == 1 and d + 1 == target:
                            results.append({
                                "row": r, "col": c,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {d} and {b} are consecutive; {target} could precede them on this diagonal."
                            })
                # Case 3: last is missing (d is None)
                if d is None and a is not None and b is not None:
                    if abs(b - a) == 1:
                        # a and b are consecutive; target could continue the sequence
                        if b - a == 1 and b + 1 == target:
                            results.append({
                                "row": r+2, "col": c+2,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {a} and {b} are consecutive; {target} could continue the sequence on this diagonal."
                            })
                        if a - b == 1 and b - 1 == target:
                            results.append({
                                "row": r+2, "col": c+2,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {b} and {a} are consecutive; {target} could continue the sequence on this diagonal."
                            })

    # Check anti-diagonals (down-left direction) similarly
    for r in range(rows - 2):
        for c in range(2, cols):
            a = board[r][c]           # first in anti-diagonal triple
            b = board[r+1][c-1]       # second (middle) in anti-diagonal
            d = board[r+2][c-2]       # third in anti-diagonal
            if [a, b, d].count(None) == 1:
                # Case 1: middle missing
                if b is None and a is not None and d is not None:
                    if abs(d - a) == 2:
                        mid_val = (a + d) // 2
                        if mid_val == target:
                            results.append({
                                "row": r+1, "col": c-1,
                                "confidence": 0.95,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {a} and {d} on an anti-diagonal are consecutive around missing {mid_val} (target)."
                            })
                # Case 2: first missing
                if a is None and b is not None and d is not None:
                    if abs(d - b) == 1:
                        if d - b == 1 and b - 1 == target:
                            results.append({
                                "row": r, "col": c,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {b} and {d} are consecutive; {target} could start the sequence on this diagonal."
                            })
                        if b - d == 1 and d + 1 == target:
                            results.append({
                                "row": r, "col": c,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {d} and {b} are consecutive; {target} could start the sequence on this diagonal."
                            })
                # Case 3: last missing
                if d is None and a is not None and b is not None:
                    if abs(b - a) == 1:
                        if b - a == 1 and b + 1 == target:
                            results.append({
                                "row": r+2, "col": c-2,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {a} and {b} are consecutive; {target} could extend the sequence on this anti-diagonal."
                            })
                        if a - b == 1 and b - 1 == target:
                            results.append({
                                "row": r+2, "col": c-2,
                                "confidence": 0.5,
                                "module": "Brain3",
                                "reason": f"Diagonal sequence: {b} and {a} are consecutive; {target} could extend the sequence on this anti-diagonal."
                            })
    return results