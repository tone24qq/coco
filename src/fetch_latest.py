"""Fetch latest draws from multiple sources with source-specific parsing."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx

HEALTH_ERROR_MARKERS = ["DB Error", "service unavailable", "系統維護"]


class FetchClient(Protocol):
    def get(self, url: str, timeout: float): ...


@dataclass
class FetchConfig:
    timeout_seconds: float
    retries: int
    backoff_seconds: float


def _check_health(text: str, source: str) -> None:
    lower = text.lower()
    for marker in HEALTH_ERROR_MARKERS:
        if marker.lower() in lower:
            raise ValueError(f"{source}: health check failed ({marker})")


def _extract_latest_issue_hint(text: str) -> Optional[str]:
    m = re.search(r"(\d{9})", text)
    return m.group(1) if m else None


def _parse_draw_rows(html: str) -> List[Dict[str, object]]:
    pattern = re.compile(
        r"(\d{9}).{0,120}?((?:\d{1,2}[,\s]+){19}\d{1,2})",
        flags=re.DOTALL,
    )
    rows: List[Dict[str, object]] = []
    for issue, nums in pattern.findall(html):
        parsed = [int(x) for x in re.split(r"[,\s]+", nums.strip()) if x]
        if len(parsed) == 20:
            rows.append({"issue": str(issue), "draw_time": "", "numbers": parsed})
    return rows


def _parse_by_source(source_name: str, html: str) -> List[Dict[str, object]]:
    if source_name in {"winwin", "pilio", "taiwanlottery"}:
        return _parse_draw_rows(html)
    return _parse_draw_rows(html)


def _fetch_winwin_api(
    client: FetchClient, timeout_seconds: float
) -> List[Dict[str, object]]:
    base_url = "https://winwin.tw/Bingo/GetBingoData?date={date_value}"
    dates_to_try = [
        date.today(),
        date.today() - timedelta(days=1),
        date.today() - timedelta(days=2),
    ]

    for d in dates_to_try:
        response = client.get(
            base_url.format(date_value=d.isoformat()), timeout=timeout_seconds
        )
        if response.status_code != 200:
            continue
        payload = response.json()
        if not isinstance(payload, list):
            continue
        records: List[Dict[str, object]] = []
        for item in payload:
            issue = str(item.get("Issue", ""))
            draw_time = str(item.get("DrawTime", ""))
            numbers = item.get("BingoNumbers", [])
            if issue and isinstance(numbers, list) and len(numbers) == 20:
                parsed_numbers = [int(x) for x in numbers]
                records.append(
                    {"issue": issue, "draw_time": draw_time, "numbers": parsed_numbers}
                )
        if records:
            return records
    return []


def fetch_latest(
    sources: List[Dict[str, str]],
    config: FetchConfig,
    client: Optional[FetchClient] = None,
) -> Tuple[List[Dict[str, object]], str, List[Dict[str, object]]]:
    attempts: List[Dict[str, object]] = []
    owns_client = client is None
    active_client: Any = client or httpx.Client(follow_redirects=True)

    try:
        for source in sources:
            source_name = source.get("name", "unknown")
            source_url = source.get("url")
            if not source_url:
                raise ValueError("Source config missing url")

            for attempt in range(1, config.retries + 2):
                try:
                    records: List[Dict[str, object]] = []
                    if source_name == "winwin":
                        records = _fetch_winwin_api(
                            active_client, config.timeout_seconds
                        )

                    if not records:
                        response = active_client.get(
                            source_url, timeout=config.timeout_seconds
                        )
                        response.raise_for_status()
                        html = response.text
                        _check_health(html, source_name)
                        records = _parse_by_source(source_name, html)

                        latest_hint = _extract_latest_issue_hint(html)
                        if records and latest_hint:
                            parsed_latest = sorted(
                                records, key=lambda x: int(str(x["issue"]))
                            )[-1]["issue"]
                            if str(parsed_latest) != str(latest_hint):
                                raise ValueError(
                                    f"{source_name}: latest hint mismatch "
                                    f"(hint={latest_hint}, parsed={parsed_latest})"
                                )

                    if not records:
                        raise ValueError(
                            f"{source_name}: parser found no valid draw rows"
                        )

                    ordered = sorted(records, key=lambda x: int(str(x["issue"])))
                    for left, right in zip(ordered, ordered[1:]):
                        if int(str(right["issue"])) - int(str(left["issue"])) != 1:
                            raise ValueError(
                                f"{source_name}: issues are not consecutive"
                            )

                    attempts.append(
                        {"source": source_name, "attempt": attempt, "status": "ok"}
                    )
                    return ordered, source_url, attempts
                except Exception as exc:  # noqa: BLE001
                    attempts.append(
                        {
                            "source": source_name,
                            "attempt": attempt,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    if attempt < config.retries + 1:
                        time.sleep(config.backoff_seconds)

        raise RuntimeError(f"All sources failed. attempts={attempts}")
    finally:
        if owns_client and isinstance(active_client, httpx.Client):
            active_client.close()
