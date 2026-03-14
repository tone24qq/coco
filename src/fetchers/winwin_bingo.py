from __future__ import annotations

import re
from typing import Any


def parse_winwin_bingo_rows(html: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    row_pattern = re.compile(r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.S | re.I)
    for match in row_pattern.finditer(html):
        row = match.group("row")
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)
        if not cells:
            continue
        plain_cells = [re.sub(r"<[^>]+>", " ", c) for c in cells]
        issue_match = re.search(r"(\d{6,12})", " ".join(plain_cells))
        if not issue_match:
            continue
        issue = int(issue_match.group(1))
        draw_time_match = re.search(
            r"(\d{1,2}:\d{2}(?::\d{2})?)", " ".join(plain_cells)
        )
        numbers_text = plain_cells[1] if len(plain_cells) > 1 else ""
        numbers = [int(x) for x in re.findall(r"\b\d{1,2}\b", numbers_text)]
        numbers = [x for x in numbers if 1 <= x <= 80]
        if len(numbers) < 20:
            continue
        numbers = numbers[:20]

        streak_match = re.search(r"連莊\D*(\d+)", " ".join(plain_cells))
        size_match = re.search(r"[大小]", " ".join(plain_cells))
        odd_even_match = re.search(r"[單雙]", " ".join(plain_cells))

        big_count = sum(1 for n in numbers if n >= 41)
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        records.append(
            {
                "issue": issue,
                "draw_time": draw_time_match.group(1) if draw_time_match else None,
                "numbers": sorted(numbers),
                "streak_count": int(streak_match.group(1)) if streak_match else None,
                "size_label": (
                    size_match.group(0)
                    if size_match
                    else ("大" if big_count >= 10 else "小")
                ),
                "odd_even_label": (
                    odd_even_match.group(0)
                    if odd_even_match
                    else ("單" if odd_count >= 10 else "雙")
                ),
            }
        )
    return records


def extract_latest_issue_hint_winwin(html: str) -> int | None:
    matches = re.findall(r"\b(\d{6,12})\b", html)
    if not matches:
        return None
    return max(int(x) for x in matches)
