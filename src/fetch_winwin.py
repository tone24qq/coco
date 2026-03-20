from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.utils import DataContractError, DrawRecord, ensure_numbers, parse_date, write_processed

WINWIN_URL = "https://winwin.tw/Bingo"
AUZO_URL = "https://lotto.auzo.tw/RK.php"
ISSUE_PATTERN = re.compile(r'"issue"\s*:\s*"?(\d+)"?')
DATE_PATTERN = re.compile(r'"draw_date"\s*:\s*"([0-9\-/]+)"')
NUMBERS_PATTERN = re.compile(r'"numbers"\s*:\s*\[(.*?)\]')


@dataclass
class FetchResult:
    records: list[DrawRecord]
    source_url: str
    fetched_at: str
    attempts: int
    failover_reason: str | None = None


def _write_debug_snapshot(html: str, source: str, reason: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("reports/fetch_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_source = re.sub(r"[^a-zA-Z0-9]+", "_", source)
    payload = {"source": source, "reason": reason, "captured_at": ts, "html": html[:200000]}
    (out_dir / f"{ts}_{safe_source}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_from_html_table(html: str) -> list[dict]:
    rows: list[dict] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(clean) < 22:
            continue
        issue = clean[0]
        draw_date = clean[1]
        nums: list[int] = []
        for c in clean[2:]:
            if c.isdigit():
                nums.append(int(c))
            if len(nums) == 20:
                break
        if issue.isdigit() and len(nums) == 20:
            rows.append({"issue": issue, "draw_date": draw_date, "numbers": nums})
    return rows


def _extract_from_json_objects(html: str) -> list[dict]:
    candidates: list[dict] = []
    for blob in re.findall(r"\{[^{}]*\}", html):
        if '"issue"' not in blob or '"numbers"' not in blob:
            continue
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if {"issue", "draw_date", "numbers"}.issubset(obj.keys()):
            candidates.append(obj)
    return candidates


def _extract_by_regex(html: str) -> list[dict]:
    issues = ISSUE_PATTERN.findall(html)
    dates = DATE_PATTERN.findall(html)
    numbers = NUMBERS_PATTERN.findall(html)
    if not (issues and dates and numbers):
        return []
    size = min(len(issues), len(dates), len(numbers))
    rows = []
    for i in range(size):
        nums = [int(x.strip()) for x in numbers[i].split(",") if x.strip().isdigit()]
        rows.append({"issue": issues[i], "draw_date": dates[i], "numbers": nums})
    return rows


def _to_draw_records(payload: list[dict]) -> list[DrawRecord]:
    if not payload:
        raise DataContractError("winwin parser failed: empty payload")
    enriched: list[tuple[str, str, tuple[int, ...]]] = []
    for item in payload:
        issue = str(item.get("issue") or "").strip()
        date_text = str(item.get("draw_date") or "").strip()
        nums = item.get("numbers")
        if not issue or not date_text or not isinstance(nums, list):
            continue
        draw_date = parse_date(date_text).isoformat()
        enriched.append((draw_date, issue, ensure_numbers(nums)))
    if not enriched:
        raise DataContractError("winwin parser failed: no valid draw rows")

    enriched.sort(key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]))
    latest_day = enriched[-1][0]
    todays = [x for x in enriched if x[0] == latest_day]
    if not todays:
        raise DataContractError("winwin parser failed: unable to isolate latest-day draws")

    results: list[DrawRecord] = []
    for idx, (draw_date, issue, numbers) in enumerate(todays, 1):
        results.append(
            DrawRecord(
                issue=issue,
                draw_date=parse_date(draw_date),
                numbers=numbers,
                day_issue_index=idx,
            )
        )
    return results


def parse_winwin_html(html: str) -> list[DrawRecord]:
    methods = [_extract_from_html_table, _extract_from_json_objects, _extract_by_regex]
    errors: list[str] = []
    for fn in methods:
        try:
            payload = fn(html)
            if payload:
                return _to_draw_records(payload)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {exc}")
    raise DataContractError(f"winwin parser failed after all strategies: {' | '.join(errors) or 'no matches'}")


def fetch_latest(sources: list[str] | None = None, timeout_s: float = 10.0) -> FetchResult:
    srcs = sources or [WINWIN_URL, AUZO_URL]
    last_err = ""
    failover_reason: str | None = None
    for idx, source in enumerate(srcs, 1):
        try:
            response = httpx.get(source, timeout=timeout_s)
            response.raise_for_status()
            rows = parse_winwin_html(response.text)
            if not rows:
                raise DataContractError("latest-day rows empty")
            if idx > 1 and failover_reason is None:
                failover_reason = "primary_source_failed_then_switched"
            return FetchResult(rows, source, datetime.now(timezone.utc).isoformat(timespec="seconds"), idx, failover_reason=failover_reason)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_err = f"transport error @ {source}: {exc}"
            _write_debug_snapshot("", source, last_err)
            if idx == 1 and len(srcs) > 1:
                failover_reason = last_err
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = f"parse error @ {source}: {exc}"
            if "response" in locals():
                _write_debug_snapshot(response.text, source, last_err)
            if idx == 1 and len(srcs) > 1:
                failover_reason = last_err
            continue
    raise DataContractError(f"fetch failed for all sources: {last_err}")


def main(output_path: str = "data/raw/winwin_latest_processed.csv") -> None:
    result = fetch_latest()
    write_processed(Path(output_path), result.records)


if __name__ == "__main__":
    main()
