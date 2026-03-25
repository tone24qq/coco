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
DIVERGENT_GAP_THRESHOLD = 5


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


def _parse_auzo_html(html: str) -> List[Dict[str, object]]:
    rows = _parse_issue_numbers_pairs(html)
    if rows:
        return _rows_to_records(rows)

    payload_pattern = re.compile(
        r'"(?:Issue|issue|drawIssue)"\s*:\s*"?(\d{9})"?.{0,200}?'
        r'"(?:numbers|BigShowOrder|bingoNumbers)"\s*:\s*\[([^\]]+)\]',
        flags=re.DOTALL,
    )
    parsed_rows: List[Tuple[str, List[int]]] = []
    for issue, nums_text in payload_pattern.findall(html):
        nums = [int(x) for x in re.split(r"[^0-9]+", nums_text.strip()) if x]
        parsed = _parse_big_show_order(nums)
        if parsed is not None:
            parsed_rows.append((str(issue), parsed))

    return _rows_to_records(parsed_rows)


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


def _build_source_tail_map(
    source_results: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, List[int]]]:
    per_source: Dict[str, Dict[str, List[int]]] = {}
    for item in source_results:
        source = str(item["source"])
        tail_records = item["records"]
        issue_map: Dict[str, List[int]] = {}
        for record in tail_records:  # type: ignore[union-attr]
            issue_map[str(record["issue"])] = [int(x) for x in record["numbers"]]
        per_source[source] = issue_map
    return per_source


def _evaluate_source_consensus(
    source_results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    if not source_results:
        return {
            "consensus_status": "divergent",
            "conflicts": [],
            "latest_issue_gap": None,
            "selected_source_reason": "no successful source",
        }

    latest_values = [int(item["latest_issue"]) for item in source_results]
    max_issue = max(latest_values)
    min_issue = min(latest_values)
    tails = _build_source_tail_map(source_results)

    conflicts: List[Dict[str, object]] = []
    issues_union = sorted(
        {int(issue) for issue_map in tails.values() for issue in issue_map.keys()}
    )
    for issue_int in issues_union:
        issue = str(issue_int)
        observed = {
            source: issue_map[issue]
            for source, issue_map in tails.items()
            if issue in issue_map
        }
        if len(observed) < 2:
            continue
        unique_numbers = {tuple(nums) for nums in observed.values()}
        if len(unique_numbers) > 1:
            conflicts.append(
                {
                    "issue": issue,
                    "sources": sorted(observed.keys()),
                    "type": "numbers_conflict",
                }
            )

    if conflicts:
        consensus_status = "divergent"
    elif max_issue == min_issue:
        consensus_status = "unanimous"
    elif max_issue - min_issue >= DIVERGENT_GAP_THRESHOLD:
        consensus_status = "divergent"
    else:
        consensus_status = "partial"

    return {
        "consensus_status": consensus_status,
        "conflicts": conflicts,
        "latest_issue_gap": max_issue - min_issue,
        "max_observed_issue": str(max_issue),
    }


def _select_best_source(
    source_results: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], str]:
    ordered = sorted(
        source_results,
        key=lambda item: (
            -int(item["latest_issue"]),
            -int(item["records_count"]),
            str(item["source"]),
        ),
    )
    selected = ordered[0]
    tie_pool = [
        item
        for item in ordered
        if int(item["latest_issue"]) == int(selected["latest_issue"])
    ]
    if len(tie_pool) == 1:
        return selected, "selected highest latest_issue"

    same_count = [
        item
        for item in tie_pool
        if int(item["records_count"]) == int(selected["records_count"])
    ]
    if len(same_count) == 1:
        return selected, "selected by latest_issue tie-break with longest records tail"
    return selected, "selected by deterministic source-name tie-break"


def fetch_latest(
    sources: List[Dict[str, str]],
    config: FetchConfig,
    client: Optional[FetchClient] = None,
) -> Tuple[List[Dict[str, object]], str, List[Dict[str, object]], Dict[str, object]]:
    attempts: List[Dict[str, object]] = []
    source_results: List[Dict[str, object]] = []
    owns_client = client is None
    active_client: Any = client or httpx.Client(follow_redirects=True)

    try:
        for source in sources:
            source_name = source.get("name", "unknown")
            source_url = source.get("url")
            if not source_url:
                raise ValueError("Source config missing url")

            source_ok = False
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
                        elif source_name == "auzo":
                            records = _parse_auzo_html(html_text)
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

                    latest_issue = str(tail[-1]["issue"])
                    attempts.append(
                        {
                            "source": source_name,
                            "attempt": attempt,
                            "status": "ok",
                            "parser_path": parser_path,
                            "latest_issue": latest_issue,
                            "records_count": len(tail),
                        }
                    )
                    source_results.append(
                        {
                            "source": source_name,
                            "url": source_url,
                            "records": tail,
                            "latest_issue": latest_issue,
                            "records_count": len(tail),
                            "parser_path": parser_path,
                            "attempt": attempt,
                        }
                    )
                    source_ok = True
                    break
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
            if not source_ok:
                continue

        if not source_results:
            raise RuntimeError(
                "All sources failed. attempts="
                + json.dumps(attempts, ensure_ascii=False, sort_keys=True)
            )

        selected, selected_reason = _select_best_source(source_results)
        consensus = _evaluate_source_consensus(source_results)

        source_latest_issues = {
            str(item["source"]): str(item["latest_issue"]) for item in source_results
        }
        source_records_count = {
            str(item["source"]): int(item["records_count"]) for item in source_results
        }

        if int(consensus.get("latest_issue_gap") or 0) >= DIVERGENT_GAP_THRESHOLD:
            selected_reason = (
                f"{selected_reason}; divergent latest gap="
                f"{consensus['latest_issue_gap']}"
            )

        diagnostics = {
            "source_latest_issues": source_latest_issues,
            "source_records_count": source_records_count,
            "selected_source_reason": selected_reason,
            "consensus_status": str(consensus["consensus_status"]),
            "max_observed_issue": str(consensus.get("max_observed_issue")),
            "source_consensus": {
                "latest_issue_gap": consensus.get("latest_issue_gap"),
                "conflicts": consensus.get("conflicts", []),
            },
        }

        return selected["records"], str(selected["url"]), attempts, diagnostics
    finally:
        if owns_client and isinstance(active_client, httpx.Client):
            active_client.close()
