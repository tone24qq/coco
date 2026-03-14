from __future__ import annotations

import re
from typing import Any

ISSUE_CONTEXT_PATTERNS = [
    re.compile(r"(?:期別|期數|開獎期別|開獎期數)\s*[:：]?\s*(\d{6,12})", re.I),
    re.compile(r"(\d{6,12})\s*(?:期別|期數)", re.I),
]


def _extract_numbers_from_cells(cells: list[str]) -> list[int]:
    candidates: list[list[int]] = []
    for cell in cells:
        plain = re.sub(r"<[^>]+>", " ", cell)
        nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", plain)]
        nums = [x for x in nums if 1 <= x <= 80]
        if len(nums) >= 20:
            candidates.append(nums[:20])

    if candidates:
        return max(candidates, key=lambda item: len(set(item)))

    whole_row = re.sub(r"<[^>]+>", " ", " ".join(cells))
    row_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", whole_row)]
    row_nums = [x for x in row_nums if 1 <= x <= 80]
    return row_nums[:20] if len(row_nums) >= 20 else []


def parse_winwin_bingo_rows(html: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    row_pattern = re.compile(r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.S | re.I)
    for match in row_pattern.finditer(html):
        row = match.group("row")
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)
        if not cells:
            continue
        plain_cells = [re.sub(r"<[^>]+>", " ", c) for c in cells]
        joined = " ".join(plain_cells)
        issue_match = re.search(r"(\d{6,12})", joined)
        if not issue_match:
            continue
        issue = int(issue_match.group(1))
        draw_time_match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", joined)
        numbers = _extract_numbers_from_cells(cells)
        if len(numbers) != 20 or len(set(numbers)) != 20:
            continue

        streak_match = re.search(r"連莊\D*(\d+)", joined)
        size_match = re.search(r"[大小]", joined)
        odd_even_match = re.search(r"[單雙]", joined)

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
    parsed_rows = parse_winwin_bingo_rows(html)
    if parsed_rows:
        return max(int(row["issue"]) for row in parsed_rows)

    contextual_candidates: list[int] = []
    for pattern in ISSUE_CONTEXT_PATTERNS:
        contextual_candidates.extend(int(x) for x in pattern.findall(html))

    if not contextual_candidates:
        return None
    return max(contextual_candidates)
