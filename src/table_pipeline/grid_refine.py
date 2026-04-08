from __future__ import annotations

import numpy as np


def merge_close_lines(lines: list[int], min_gap: int) -> list[int]:
    if not lines:
        return []
    lines = sorted(int(x) for x in lines)
    merged: list[list[int]] = [[lines[0]]]
    for p in lines[1:]:
        if p - merged[-1][-1] <= min_gap:
            merged[-1].append(p)
        else:
            merged.append([p])
    return [int(round(float(np.mean(g)))) for g in merged]


def repair_line_sequence(lines: list[int], limit: int, expected_count: int | None = None) -> tuple[list[int], str]:
    lines = [x for x in lines if 0 <= x <= limit]
    if not lines:
        return [0, limit], "fallback_even_split"
    lines = merge_close_lines(lines, max(2, limit // 160))
    if lines[0] > limit * 0.05:
        lines.insert(0, 0)
    else:
        lines[0] = 0
    if lines[-1] < limit * 0.95:
        lines.append(limit)
    else:
        lines[-1] = limit

    source = "detected_lines"
    if expected_count is not None and len(lines) != expected_count:
        source = "repaired_lines"
        if len(lines) < expected_count:
            arr = np.array(lines, dtype=np.int32)
            while len(arr) < expected_count:
                gaps = np.diff(arr)
                idx = int(np.argmax(gaps))
                mid = int((arr[idx] + arr[idx + 1]) // 2)
                arr = np.insert(arr, idx + 1, mid)
            lines = arr.tolist()
        elif len(lines) > expected_count:
            arr = np.array(lines, dtype=np.int32)
            while len(arr) > expected_count:
                gaps = np.diff(arr)
                idx = int(np.argmin(gaps))
                arr = np.delete(arr, idx + 1)
            lines = arr.tolist()
    return lines, source
