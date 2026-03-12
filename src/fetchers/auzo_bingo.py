from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCES = [
    "https://lotto.auzo.tw/RI.php",
    "https://lotto.auzo.tw/RL.php",
]


class FetchDrawsError(ValueError):
    """Raised when remote draw data cannot be fetched or validated."""


@dataclass(frozen=True)
class DrawRecord:
    issue: int
    draw_time: str | None
    numbers: list[int]


class BingoDrawFetcher:
    def __init__(
        self,
        sources: list[str] | None = None,
        timeout: float = 8.0,
        retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ):
        self.sources = sources or DEFAULT_SOURCES
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def fetch_recent_records(
        self, min_draws: int, max_draws: int
    ) -> tuple[list[DrawRecord], str]:
        errors: list[str] = []
        for source in self.sources:
            try:
                records = self._fetch_from_source(
                    source, min_draws=min_draws, max_draws=max_draws
                )
                return records, source
            except FetchDrawsError as exc:
                errors.append(f"{source}: {exc}")
        raise FetchDrawsError("all sources failed: " + " | ".join(errors))

    def _fetch_from_source(
        self, source: str, min_draws: int, max_draws: int
    ) -> list[DrawRecord]:
        html = self._fetch_html(source)
        self._check_source_health(html)
        parsed = self._parse_records_by_source(source, html)
        if not parsed:
            raise FetchDrawsError("cannot parse any valid draw rows")

        deduped: dict[int, DrawRecord] = {}
        for row in parsed:
            normalized = self._normalize_row(row)
            existing = deduped.get(normalized.issue)
            if existing is not None and existing != normalized:
                raise FetchDrawsError(
                    f"duplicate issue {normalized.issue} contains conflicting content"
                )
            deduped[normalized.issue] = normalized

        ordered = sorted(deduped.values(), key=lambda item: item.issue)
        if len(ordered) < min_draws:
            raise FetchDrawsError(
                f"valid records {len(ordered)} below minimum required {min_draws}"
            )

        latest_issue_hint = self._extract_latest_issue_hint_by_source(source, html)
        if latest_issue_hint is not None and ordered[-1].issue != latest_issue_hint:
            raise FetchDrawsError(
                "latest issue mismatch between page hint "
                f"({latest_issue_hint}) and parsed records ({ordered[-1].issue})"
            )

        selected = ordered[-min(max_draws, len(ordered)) :]
        self._ensure_consecutive_issues(selected)

        LOGGER.info(
            "source=%s parsed_total=%d deduped_total=%d sorted_issue_range=%s-%s selected_issue_range=%s-%s",
            source,
            len(parsed),
            len(ordered),
            ordered[0].issue,
            ordered[-1].issue,
            selected[0].issue,
            selected[-1].issue,
        )
        return selected

    def _fetch_html(self, source: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with request.urlopen(source, timeout=self.timeout) as resp:
                    raw = resp.read()
                return raw.decode("utf-8", errors="ignore")
            except (error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise FetchDrawsError(f"fetch failed after retry: {last_error}")

    def _check_source_health(self, html: str) -> None:
        lowered = html.lower()
        if "db error" in lowered or "database error" in lowered:
            raise FetchDrawsError("source returned DB Error")
        if "service unavailable" in lowered:
            raise FetchDrawsError("source returned service unavailable")

    def _parse_records_by_source(self, source: str, html: str) -> list[dict[str, Any]]:
        parser = self._choose_source_parser(source)
        return parser(html)

    def _extract_latest_issue_hint_by_source(
        self, source: str, html: str
    ) -> int | None:
        hostname = urlparse(source).netloc.lower()
        if "auzo.tw" in hostname:
            return self._extract_latest_issue_hint_auzo(html)
        return None

    def _choose_source_parser(
        self, source: str
    ) -> Callable[[str], list[dict[str, Any]]]:
        path = urlparse(source).path.lower()
        if path.endswith("/ri.php"):
            return self._parse_auzo_ri
        if path.endswith("/rl.php"):
            return self._parse_auzo_rl
        hostname = urlparse(source).netloc.lower()
        if "auzo.tw" in hostname:
            return self._parse_auzo_generic
        raise FetchDrawsError(f"unsupported source parser for url: {source}")

    def _parse_auzo_ri(self, html: str) -> list[dict[str, Any]]:
        return self._parse_auzo_generic(html)

    def _parse_auzo_rl(self, html: str) -> list[dict[str, Any]]:
        return self._parse_auzo_generic(html)

    def _parse_auzo_generic(self, html: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        pattern = re.compile(
            r"(?P<issue>\d{6,12}).{0,220}?"
            r"(?P<numbers>(?:\b\d{1,2}\b[\s,|、]+){19}\b\d{1,2}\b)",
            flags=re.S,
        )
        for match in pattern.finditer(html):
            issue = int(match.group("issue"))
            around = html[
                max(0, match.start() - 200) : min(len(html), match.end() + 200)
            ]
            draw_time_match = re.search(
                r"(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)",
                around,
            )
            draw_time = draw_time_match.group(1) if draw_time_match else None
            numbers = [int(x) for x in re.findall(r"\d{1,2}", match.group("numbers"))]
            records.append({"issue": issue, "draw_time": draw_time, "numbers": numbers})
        return records

    def _extract_latest_issue_hint_auzo(self, html: str) -> int | None:
        markers = [
            r"最新(?:一期|期數)[^\d]{0,8}(\d{6,12})",
            r"第\s*(\d{6,12})\s*期",
        ]
        candidates: list[int] = []
        for pattern in markers:
            for matched in re.findall(pattern, html):
                candidates.append(int(matched))
        return max(candidates) if candidates else None

    def _normalize_row(self, row: dict[str, Any]) -> DrawRecord:
        issue = int(row["issue"])
        draw_time = row.get("draw_time")
        numbers = row.get("numbers")
        if not isinstance(numbers, list):
            raise FetchDrawsError(f"issue {issue} missing numbers")
        if len(numbers) != 20:
            raise FetchDrawsError(f"issue {issue} must contain exactly 20 numbers")
        if any((not isinstance(n, int)) for n in numbers):
            raise FetchDrawsError(f"issue {issue} has non-integer numbers")
        if any(n < 1 or n > 80 for n in numbers):
            raise FetchDrawsError(f"issue {issue} contains numbers out of range 1-80")
        if len(set(numbers)) != 20:
            raise FetchDrawsError(f"issue {issue} contains duplicate numbers")
        return DrawRecord(issue=issue, draw_time=draw_time, numbers=sorted(numbers))

    def _ensure_consecutive_issues(self, records: list[DrawRecord]) -> None:
        for prev, cur in zip(records, records[1:]):
            if cur.issue != prev.issue + 1:
                raise FetchDrawsError(
                    f"issues are not consecutive: {prev.issue} -> {cur.issue}"
                )


def build_recent_draws(
    fetcher: BingoDrawFetcher, min_draws: int, max_draws: int
) -> tuple[list[list[int]], list[DrawRecord], str]:
    records, source = fetcher.fetch_recent_records(
        min_draws=min_draws, max_draws=max_draws
    )
    recent_draws = [record.numbers for record in records]
    if len(recent_draws) != len(records):
        raise FetchDrawsError("records and recent_draws length mismatch")

    LOGGER.info(
        "auto fetch complete source=%s total=%d first_issue=%s last_issue=%s",
        source,
        len(records),
        records[0].issue,
        records[-1].issue,
    )
    return recent_draws, records, source
