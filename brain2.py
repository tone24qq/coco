def analyze(board, target):
    """
    Brain2: Skip Pattern Analysis.
    Looks for identical numbers that appear with exactly one row/column gap between them.
    If the target fits such a repeating pattern (appearing every other row/col), flag the gap.
    Also flags gaps where another number repeats (to potentially rule out target).
    """
    results = []
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0

    # Vertical skip pattern: same number appears in a column with one row gap
    for c in range(cols):
        for r in range(rows - 2):
            if (board[r][c] is not None and board[r+2][c] is not None 
                    and board[r][c] == board[r+2][c] 
                    and board[r+1][c] is None):
                val = board[r][c]
                if val == target:
                    # Target number appears two rows apart in the same column, likely also in the gap
                    results.append({
                        "row": r+1, "col": c,
                        "confidence": 0.9,
                        "module": "Brain2",
                        "reason": f"Skip pattern: {target} found in col {c} at rows {r} and {r+2}, so row {r+1}, col {c} is likely {target}."
                    })
                else:
                    # Another number repeats every other row here, so the gap is probably that number (not the target)
                    results.append({
                        "row": r+1, "col": c,
                        "confidence": 0.0,
                        "module": "Brain2",
                        "reason": f"Skip pattern: Number {val} repeats in col {c} (rows {r} and {r+2}) skipping one row, so hidden cell at row {r+1} is likely {val}, not {target}."
                    })

    # Horizontal skip pattern: same number appears in a row with one column gap
    for r in range(rows):
        for c in range(cols - 2):
            if (board[r][c] is not None and board[r][c+2] is not None 
                    and board[r][c] == board[r][c+2] 
                    and board[r][c+1] is None):
                val = board[r][c]
                if val == target:
                    results.append({
                        "row": r, "col": c+1,
                        "confidence": 0.9,
                        "module": "Brain2",
                        "reason": f"Skip pattern: {target} found in row {r} at cols {c} and {c+2}, so col {c+1}, row {r} is likely {target}."
                    })
                else:
                    results.append({
                        "row": r, "col": c+1,
                        "confidence": 0.0,
                        "module": "Brain2",
                        "reason": f"Skip pattern: Number {val} repeats in row {r} (cols {c} and {c+2}) skipping one column, so hidden col {c+1} is likely {val}, not {target}."
                    })
    return results