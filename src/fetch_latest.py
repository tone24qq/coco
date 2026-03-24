"""Fetch latest draws from multiple sources with source-specific parsing."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import httpx

HEALTH_ERROR_MARKERS = ["DB Error", "service unavailable", "系統維護"]
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/132.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.google.com/",
}
MIN_CONSECUTIVE_TAIL = 5
HINT_MISMATCH_TOLERANCE = 200


class FetchClient(Protocol):
    def get(
        self,
        url: str,
        timeout: float,
        headers: Optional[Dict[str, str]] = None,
    ): ...


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
    matches = re.findall(r"(\d{9})", text)
    return max(matches, key=int) if matches else None


def _parse_big_show_order(raw: object) -> Optional[List[int]]:
    if isinstance(raw, list):
        values = [int(x) for x in raw]
    elif isinstance(raw, str):
        values = [int(x) for x in re.split(r"[^0-9]+", raw.strip()) if x]
    else:
        return None

    if len(values) != 20:
        return None
    if len(set(values)) != 20:
        return None
    if any(v < 1 or v > 80 for v in values):
        return None
    return values


def _parse_issue_numbers_pairs(text: str) -> List[Tuple[str, List[int]]]:
    pattern = re.compile(
        r"(\d{9}).{0,200}?((?:\d{1,2}[,\s、]+){19}\d{1,2})",
        flags=re.DOTALL,
    )
    rows: List[Tuple[str, List[int]]] = []
    for issue, nums in pattern.findall(text):
        parsed = [int(x) for x in re.split(r"[,\s、]+", nums.strip()) if x]
        parsed_valid = _parse_big_show_order(parsed)
        if parsed_valid is not None:
            rows.append((str(issue), parsed_valid))
    return rows


def _rows_to_records(rows: Sequence[Tuple[str, List[int]]]) -> List[Dict[str, object]]:
    return [
        {"issue": str(issue), "draw_time": "", "numbers": list(numbers)}
        for issue, numbers in rows
    ]


def _parse_winwin_json_payload(payload: object) -> List[Dict[str, object]]:
    if not isinstance(payload, list):
        return []

    records: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        issue_raw = item.get("Issue", item.get("No", ""))
        draw_time_raw = item.get("DrawTime", item.get("OpenDate", ""))
        numbers_raw = item.get("BingoNumbers", item.get("BigShowOrder"))

        issue = str(issue_raw).strip()
        if not issue or not re.fullmatch(r"\d{9}", issue):
            continue

        numbers = _parse_big_show_order(numbers_raw)
        if numbers is None:
            continue

        records.append(
            {
                "issue": issue,
                "draw_time": str(draw_time_raw or "").strip(),
                "numbers": numbers,
            }
        )

    return records


def _parse_winwin_html(html: str) -> List[Dict[str, object]]:
    return _rows_to_records(_parse_issue_numbers_pairs(html))


def _parse_pilio_html(html: str) -> List[Dict[str, object]]:
    return _rows_to_records(_parse_issue_numbers_pairs(html))


def _parse_taiwanlottery_html(html: str) -> List[Dict[str, object]]:
    return _rows_to_records(_parse_issue_numbers_pairs(html))


def _parse_generic_html(html: str) -> List[Dict[str, object]]:
    return _rows_to_records(_parse_issue_numbers_pairs(html))


def _get_with_headers(
    client: FetchClient,
    url: str,
    timeout_seconds: float,
    headers: Optional[Dict[str, str]] = None,
):
    merged = dict(DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    return client.get(url, timeout=timeout_seconds, headers=merged)


def _fetch_winwin_api(
    client: FetchClient, timeout_seconds: float
) -> Tuple[List[Dict[str, object]], str]:
    base_url = "https://winwin.tw/Bingo/GetBingoData?date={date_value}"
    dates_to_try = [
        date.today(),
        date.today() - timedelta(days=1),
        date.today() - timedelta(days=2),
    ]

    for target_date in dates_to_try:
        response = _get_with_headers(
            client=client,
            url=base_url.format(date_value=target_date.isoformat()),
            timeout_seconds=timeout_seconds,
            headers={"Accept": "application/json,text/plain,*/*"},
        )
        if response.status_code != 200:
            continue
        records = _parse_winwin_json_payload(response.json())
        if records:
            return records, "json"
    return [], "json"


def _dedupe_and_validate_conflicts(
    source_name: str, records: Sequence[Dict[str, object]]
) -> List[Dict[str, object]]:
    by_issue: Dict[int, Dict[str, object]] = {}
    for record in records:
        issue_int = int(str(record["issue"]))
        numbers = [int(x) for x in record["numbers"]]  # type: ignore[index]
        key = ",".join(str(n) for n in numbers)

        existing = by_issue.get(issue_int)
        if existing is None:
            by_issue[issue_int] = {
                "issue": str(issue_int),
                "draw_time": str(record.get("draw_time", "")),
                "numbers": numbers,
                "_numbers_key": key,
            }
            continue

        if str(existing["_numbers_key"]) != key:
            raise ValueError(f"{source_name}: conflicting duplicate issue={issue_int}")

    ordered = [by_issue[k] for k in sorted(by_issue.keys())]
    for row in ordered:
        row.pop("_numbers_key", None)
    return ordered


def _latest_consecutive_tail(
    records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not records:
        return []

    tail: List[Dict[str, object]] = [records[-1]]
    for idx in range(len(records) - 2, -1, -1):
        left_issue = int(str(records[idx]["issue"]))
        right_issue = int(str(tail[0]["issue"]))
        if right_issue - left_issue == 1:
            tail.insert(0, records[idx])
        else:
            break
    return tail


def _validate_hint(
    source_name: str,
    hint_issue: Optional[str],
    parsed_records: Sequence[Dict[str, object]],
) -> None:
    if not hint_issue or not parsed_records:
        return
    parsed_max = int(str(parsed_records[-1]["issue"]))
    hint_max = int(str(hint_issue))
    if hint_max < parsed_max:
        return
    if hint_max - parsed_max > HINT_MISMATCH_TOLERANCE:
        raise ValueError(
            f"{source_name}: latest hint mismatch too large "
            f"(hint={hint_max}, parsed={parsed_max})"
        )


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
                parser_path = "unknown"
                try:
                    records: List[Dict[str, object]] = []
                    html_text = ""

                    if source_name == "winwin":
                        records, parser_path = _fetch_winwin_api(
                            active_client, config.timeout_seconds
                        )

                    if not records:
                        response = _get_with_headers(
                            client=active_client,
                            url=source_url,
                            timeout_seconds=config.timeout_seconds,
                            headers={"Referer": source_url},
                        )
                        response.raise_for_status()
                        html_text = response.text
                        _check_health(html_text, source_name)

                        parser_path = f"{source_name}:html"
                        if source_name == "winwin":
                            records = _parse_winwin_html(html_text)
                        elif source_name == "pilio":
                            records = _parse_pilio_html(html_text)
                        elif source_name == "taiwanlottery":
                            records = _parse_taiwanlottery_html(html_text)
                        else:
                            records = []

                        if not records:
                            parser_path = "generic"
                            records = _parse_generic_html(html_text)

                    if not records:
                        raise ValueError(
                            f"{source_name}: parser found no valid draw rows"
                        )

                    deduped = _dedupe_and_validate_conflicts(source_name, records)
                    tail = _latest_consecutive_tail(deduped)
                    if len(tail) < MIN_CONSECUTIVE_TAIL:
                        raise ValueError(
                            f"{source_name}: consecutive tail too short "
                            f"(tail={len(tail)}, min={MIN_CONSECUTIVE_TAIL})"
                        )

                    hint_issue = _extract_latest_issue_hint(html_text)
                    _validate_hint(source_name, hint_issue, tail)

                    attempts.append(
                        {
                            "source": source_name,
                            "attempt": attempt,
                            "status": "ok",
                            "parser_path": parser_path,
                        }
                    )
                    return tail, source_url, attempts
                except Exception as exc:  # noqa: BLE001
                    attempts.append(
                        {
                            "source": source_name,
                            "attempt": attempt,
                            "status": "error",
                            "parser_path": parser_path,
                            "error": str(exc),
                        }
                    )
                    if attempt < config.retries + 1:
                        time.sleep(config.backoff_seconds)

        raise RuntimeError(
            "All sources failed. attempts="
            + json.dumps(attempts, ensure_ascii=False, sort_keys=True)
        )
    finally:
        if owns_client and isinstance(active_client, httpx.Client):
            active_client.close()
