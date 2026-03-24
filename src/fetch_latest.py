"""Fetch latest draw records with retries, source failover, and validation hooks."""

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


def _check_health(text: str) -> None:
    lower = text.lower()
    for marker in HEALTH_ERROR_MARKERS:
        if marker.lower() in lower:
            raise ValueError(f"Source health check failed: {marker}")


def _extract_latest_issue_hint(text: str) -> Optional[str]:
    match = re.search(r"(\d{9})", text)
    return match.group(1) if match else None


def _parse_generic_html(text: str) -> List[Dict[str, object]]:
    pattern = re.compile(r"(\d{9})[^0-9]+((?:\d{1,2}[,\s]+){19}\d{1,2})")
    records: List[Dict[str, object]] = []
    for issue, numbers_text in pattern.findall(text):
        numbers = [int(x) for x in re.split(r"[,\s]+", numbers_text.strip()) if x]
        if len(numbers) == 20:
            records.append({"issue": issue, "draw_time": "", "numbers": numbers})
    return records


def _fetch_winwin_api(
    client: FetchClient, config: FetchConfig
) -> List[Dict[str, object]]:
    base_url = "https://winwin.tw/Bingo/GetBingoData?date={date_value}"
    dates_to_try = [
        date.today(),
        date.today() - timedelta(days=1),
        date.today() - timedelta(days=2),
    ]

    for date_value in dates_to_try:
        url = base_url.format(date_value=date_value.isoformat())
        response = client.get(url, timeout=config.timeout_seconds)
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
            if issue and isinstance(numbers, list):
                records.append(
                    {"issue": issue, "draw_time": draw_time, "numbers": numbers}
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

            last_error = ""
            for attempt in range(config.retries + 1):
                try:
                    if "winwin.tw/Bingo" in source_url:
                        records = _fetch_winwin_api(active_client, config)
                        if records:
                            attempts.append(
                                {
                                    "source": source_name,
                                    "attempt": attempt + 1,
                                    "status": "ok",
                                }
                            )
                            return records, source_url, attempts

                    response = active_client.get(
                        source_url, timeout=config.timeout_seconds
                    )
                    response.raise_for_status()
                    html = response.text
                    _check_health(html)
                    records = _parse_generic_html(html)
                    if not records:
                        raise ValueError(
                            "No valid latest records found in source response"
                        )

                    latest_hint = _extract_latest_issue_hint(html)
                    parsed_latest = sorted(
                        records, key=lambda item: str(item["issue"])
                    )[-1]["issue"]
                    if latest_hint and str(latest_hint) != str(parsed_latest):
                        raise ValueError(
                            "Latest issue hint mismatch: "
                            f"hint={latest_hint}, parsed={parsed_latest}"
                        )

                    ordered = sorted(records, key=lambda item: int(str(item["issue"])))
                    for left, right in zip(ordered, ordered[1:]):
                        if int(str(right["issue"])) - int(str(left["issue"])) != 1:
                            raise ValueError("Fetched issues are not consecutive")

                    attempts.append(
                        {"source": source_name, "attempt": attempt + 1, "status": "ok"}
                    )
                    return ordered, source_url, attempts
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    attempts.append(
                        {
                            "source": source_name,
                            "attempt": attempt + 1,
                            "status": "error",
                            "error": last_error,
                        }
                    )
                    if attempt < config.retries:
                        time.sleep(config.backoff_seconds)

            if last_error:
                continue
    finally:
        if owns_client and isinstance(active_client, httpx.Client):
            active_client.close()

    raise RuntimeError(f"All sources failed. attempts={attempts}")
