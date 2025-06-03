def analyze(board, target):
    """
    Brain1: Tail Number Pattern Analysis.
    Identifies rows or columns where all revealed numbers share the same last digit.
    If the target's last digit matches that pattern, hidden cells in that line are likely the target.
    """
    results = []
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    target_last = target % 10  # last digit of target number

    # Check each row for uniform last-digit pattern
    for r in range(rows):
        known_nums = [x for x in board[r] if x is not None]
        if not known_nums:
            continue
        last_digits = {x % 10 for x in known_nums}
        if len(last_digits) == 1:  # all known numbers in this row have the same tail digit
            tail = last_digits.pop()
            if tail == target_last:
                # Every hidden cell in this row could potentially be the target (matching the tail pattern)
                for c in range(cols):
                    if board[r][c] is None:
                        results.append({
                            "row": r, "col": c,
                            "confidence": 0.8,
                            "module": "Brain1",
                            "reason": f"Tail pattern: All revealed numbers in row {r} end with '{tail}', so a hidden cell might be {target} (ends with {tail})."
                        })

    # Check each column for uniform last-digit pattern
    for c in range(cols):
        known_nums = [board[r][c] for r in range(rows) if board[r][c] is not None]
        if not known_nums:
            continue
        last_digits = {x % 10 for x in known_nums}
        if len(last_digits) == 1:
            tail = last_digits.pop()
            if tail == target_last:
                for r in range(rows):
                    if board[r][c] is None:
                        results.append({
                            "row": r, "col": c,
                            "confidence": 0.8,
                            "module": "Brain1",
                            "reason": f"Tail pattern: All revealed numbers in col {c} end with '{tail}', so a hidden cell might be {target} (ends with {tail})."
                        })

    # Global tail check: if target's last digit is completely missing among all revealed numbers
    all_revealed_tails = {x % 10 for row in board for x in row if x is not None}
    if target_last not in all_revealed_tails:
        # Target's last digit hasn't appeared in any known number, indicating target could be hidden.
        for r in range(rows):
            for c in range(cols):
                if board[r][c] is None:
                    results.append({
                        "row": r, "col": c,
                        "confidence": 0.3,
                        "module": "Brain1",
                        "reason": f"Tail pattern: The digit '{target_last}' (from target {target}) is absent among revealed numbers, suggesting a hidden occurrence."
                    })

    return results