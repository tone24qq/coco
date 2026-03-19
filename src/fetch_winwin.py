from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.utils import DataContractError, DrawRecord, ensure_numbers, parse_date, write_processed

WINWIN_URL = "https://winwin.tw/Bingo"
ISSUE_PATTERN = re.compile(r'"issue"\s*:\s*"?(\d+)"?')
DATE_PATTERN = re.compile(r'"draw_date"\s*:\s*"([0-9\-/]+)"')
NUMBERS_PATTERN = re.compile(r'"numbers"\s*:\s*\[(.*?)\]')


@dataclass
class FetchResult:
    records: list[DrawRecord]
    source_url: str
    fetched_at: str
    attempts: int


def _write_debug_snapshot(html: str, source: str, reason: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("reports/fetch_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_source = re.sub(r"[^a-zA-Z0-9]+", "_", source)
    payload = {"source": source, "reason": reason, "captured_at": ts, "html": html[:200000]}
    (out_dir / f"{ts}_{safe_source}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    rows = sorted(payload, key=lambda x: (str(x.get("draw_date", "")), str(x.get("issue", ""))))
    results: list[DrawRecord] = []
    per_day: dict[str, int] = {}
    for item in rows:
        issue = str(item.get("issue") or "").strip()
        date_text = str(item.get("draw_date") or "").strip()
        nums = item.get("numbers")
        if not issue or not date_text or not isinstance(nums, list):
            continue
        draw_date = parse_date(date_text)
        key = draw_date.isoformat()
        per_day[key] = per_day.get(key, 0) + 1
        results.append(
            DrawRecord(
                issue=issue,
                draw_date=draw_date,
                numbers=ensure_numbers(nums),
                day_issue_index=per_day[key],
            )
        )
    if not results:
        raise DataContractError("winwin parser failed: no valid draw rows")
    return results


def parse_winwin_html(html: str) -> list[DrawRecord]:
    methods = [_extract_from_json_objects, _extract_by_regex]
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
    srcs = sources or [WINWIN_URL]
    last_err = ""
    for idx, source in enumerate(srcs, 1):
        try:
            response = httpx.get(source, timeout=timeout_s)
            response.raise_for_status()
            rows = parse_winwin_html(response.text)
            return FetchResult(rows, source, datetime.now(timezone.utc).isoformat(timespec="seconds"), idx)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_err = f"transport error @ {source}: {exc}"
            _write_debug_snapshot("", source, last_err)
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = f"parse error @ {source}: {exc}"
            if 'response' in locals():
                _write_debug_snapshot(response.text, source, last_err)
            continue
    raise DataContractError(f"fetch failed for all sources: {last_err}")


def main(output_path: str = "data/raw/winwin_latest_processed.csv") -> None:
    result = fetch_latest()
    write_processed(Path(output_path), result.records)


if __name__ == "__main__":
    main()
