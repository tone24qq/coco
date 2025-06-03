def analyze(board, target):
    """
    Brain4 (NewModule): Difference Pattern Analysis.
    Finds places where two numbers in the same row or column have exactly one hidden cell between them,
    and the hidden cell would be the numeric midpoint. If the midpoint equals the target, flag that cell.
    """
    results = []
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0

    # Horizontal difference pattern: two known numbers with one gap between them
    for r in range(rows):
        for c in range(cols - 2):
            if (board[r][c] is not None and board[r][c+2] is not None 
                    and board[r][c+1] is None):
                left_val = board[r][c]
                right_val = board[r][c+2]
                # If the difference is even, there is a well-defined midpoint
                if (right_val - left_val) % 2 == 0:
                    mid_val = (left_val + right_val) // 2
                    if mid_val == target:
                        results.append({
                            "row": r, "col": c+1,
                            "confidence": 0.9,
                            "module": "Brain4",
                            "reason": f"Difference pattern: {left_val} and {right_val} in row {r} are evenly spaced around {mid_val} (target)."
                        })

    # Vertical difference pattern: two known numbers with one gap between them
    for c in range(cols):
        for r in range(rows - 2):
            if (board[r][c] is not None and board[r+2][c] is not None 
                    and board[r+1][c] is None):
                top_val = board[r][c]
                bottom_val = board[r+2][c]
                if (bottom_val - top_val) % 2 == 0:
                    mid_val = (top_val + bottom_val) // 2
                    if mid_val == target:
                        results.append({
                            "row": r+1, "col": c,
                            "confidence": 0.9,
                            "module": "Brain4",
                            "reason": f"Difference pattern: {top_val} and {bottom_val} in col {c} are evenly spaced around {mid_val} (target)."
                        })
    return results