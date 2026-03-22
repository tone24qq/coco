from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.io_utils import write_json_gz_if_needed
from src.utils import DataContractError, DrawRecord, enforce_file_size, ensure_numbers, parse_date, write_processed

WINWIN_URL = "https://winwin.tw/Bingo"
AUZO_URL = "https://lotto.auzo.tw/RK.php"
WINWIN_DYNAMIC_URL = "https://winwin.tw/Bingo/GetBingoData"

_DYNAMIC_MARKERS = (
    "loadBingoData(",
    "/Bingo/GetBingoData",
    'id="bingoTable"',
    "id='bingoTable'",
)
_DB_ERROR_MARKERS = ("db error", "database error", "service unavailable")


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


def _numbers_from_text(text: str) -> list[int]:
    nums: list[int] = []
    for token in re.findall(r"\d+", text):
        value = int(token)
        if 1 <= value <= 80:
            nums.append(value)
    return nums


def _extract_from_html_table(html: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        clean = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        if len(clean) < 3:
            continue
        issue_match = re.search(r"\d{8,}", clean[0])
        if not issue_match:
            continue
        issue = issue_match.group(0)
        date_cell = clean[1]
        nums: list[int] = []
        for c in clean[2:]:
            nums.extend(_numbers_from_text(c))
            if len(nums) >= 20:
                break
        if len(nums) >= 20:
            rows.append({"issue": issue, "draw_date": date_cell, "numbers": nums[:20]})
    return rows


def _extract_by_regex(html: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"(?:No|issue|期別)\s*[:：\"]*\s*(\d{8,}).{0,180}?"
        r"(?:OpenDate|draw_date|開獎時間|開獎日期)?\s*[:：\"]*\s*([0-9\-/ :T]{8,25})?.{0,280}?"
        r"(?:BigShowOrder|numbers|獎號|號碼)\s*[:：\[\"]*\s*([^\]<>}{]{20,200})",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for issue, date_text, nums_text in pattern.findall(html):
        nums = _numbers_from_text(nums_text)
        if len(nums) != 20:
            continue
        rows.append({"issue": issue, "draw_date": date_text or datetime.now().date().isoformat(), "numbers": nums})
    return rows


def _extract_current_date_hint(html: str) -> str | None:
    m = re.search(r"currentDate\s*[:=]\s*['\"]([0-9\-/]{8,10})['\"]", html, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).replace("/", "-")


def _parse_draw_date(raw: str) -> str:
    txt = (raw or "").strip()
    if not txt:
        return datetime.now().date().isoformat()
    txt = txt.replace("/", "-")
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}", txt)
    if iso_match:
        return iso_match.group(0)
    return parse_date(txt).isoformat()


def _to_draw_records(payload: list[dict[str, object]], *, source_name: str) -> list[DrawRecord]:
    if not payload:
        raise DataContractError(f"{source_name} parser failed: empty payload")
    enriched: list[tuple[str, str, tuple[int, ...]]] = []
    for item in payload:
        issue = str(item.get("issue") or "").strip()
        date_text = str(item.get("draw_date") or "").strip()
        nums = item.get("numbers")
        if not issue or not isinstance(nums, list):
            continue
        try:
            draw_date = _parse_draw_date(date_text)
            numbers = ensure_numbers(nums)
        except Exception:
            continue
        enriched.append((draw_date, issue, numbers))
    if not enriched:
        raise DataContractError(f"{source_name} parser failed: no valid draw rows")

    enriched.sort(key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]))
    latest_day = enriched[-1][0]
    todays = [x for x in enriched if x[0] == latest_day]
    if not todays:
        raise DataContractError(f"{source_name} parser failed: unable to isolate latest-day draws")

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
    _validate_same_day_issue_completeness(results, source_name=source_name)
    return results


def _issue_suffix_index(issue: str) -> int | None:
    if not issue.isdigit() or len(issue) < 3:
        return None
    return int(issue[-3:])


def _validate_same_day_issue_completeness(rows: list[DrawRecord], source_name: str) -> None:
    if not rows:
        raise DataContractError(f"{source_name} same-day rows empty")
    ordered = sorted(rows, key=lambda r: int(r.issue) if r.issue.isdigit() else r.issue)
    suffixes: list[int] = []
    for row in ordered:
        suffix = _issue_suffix_index(row.issue)
        if suffix is None:
            raise DataContractError(f"{source_name} issue format invalid for same-day contract: {row.issue}")
        suffixes.append(suffix)

    expected = list(range(min(suffixes), max(suffixes) + 1))
    if suffixes != expected:
        raise DataContractError(
            f"{source_name} same-day issue incomplete: expected contiguous suffix {expected[0]}..{expected[-1]} got {suffixes[:3]}...{suffixes[-3:]}"
        )
    day_idx = [row.day_issue_index for row in ordered]
    expected_idx = list(range(1, len(ordered) + 1))
    if day_idx != expected_idx:
        raise DataContractError(f"{source_name} day_issue_index contract violated")


def _has_dynamic_marker(html: str) -> bool:
    lower = html.lower()
    return any(marker.lower() in lower for marker in _DYNAMIC_MARKERS)


def _parse_winwin_static_html(html: str) -> list[DrawRecord]:
    payload = _extract_from_html_table(html)
    if payload:
        return _to_draw_records(payload, source_name="winwin")
    payload = _extract_by_regex(html)
    if payload:
        return _to_draw_records(payload, source_name="winwin")
    raise DataContractError("winwin parser failed: no static rows")


def _parse_winwin_dynamic_payload(payload: dict | list) -> list[DrawRecord]:
    if isinstance(payload, dict):
        candidates = payload.get("Data") or payload.get("data") or payload.get("rows") or payload.get("list") or []
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = []

    rows: list[dict[str, object]] = []
    for row in candidates:
        if not isinstance(row, dict):
            continue
        issue = str(row.get("No") or row.get("issue") or "").strip()
        open_date = str(row.get("OpenDate") or row.get("draw_date") or "").strip()
        big_show = row.get("BigShowOrder") or row.get("numbers")
        if isinstance(big_show, str):
            nums = [int(x) for x in re.findall(r"\d+", big_show)]
        elif isinstance(big_show, list):
            nums = [int(x) for x in big_show if str(x).strip().isdigit()]
        else:
            nums = []
        if len(nums) != 20:
            continue
        rows.append(
            {
                "issue": issue,
                "draw_date": open_date,
                "numbers": nums,
                "high_low_top": row.get("HighLowTop"),
                "odd_even_top": row.get("OddEvenTop"),
            }
        )
    return _to_draw_records(rows, source_name="winwin_dynamic")


def _date_candidates(html: str) -> list[str]:
    out: list[str] = []
    hint = _extract_current_date_hint(html)
    if hint:
        out.append(hint)
    base = datetime.now().date()
    for lag in range(3):
        out.append((base - timedelta(days=lag)).isoformat())
    dedup: list[str] = []
    for x in out:
        if x not in dedup:
            dedup.append(x)
    return dedup


def _fetch_winwin_dynamic_records(html: str, timeout_s: float) -> list[DrawRecord]:
    for date_str in _date_candidates(html):
        resp = httpx.get(WINWIN_DYNAMIC_URL, params={"date": date_str}, timeout=timeout_s)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise DataContractError(f"winwin dynamic endpoint returned non-json payload for date={date_str}") from exc
        try:
            return _parse_winwin_dynamic_payload(payload)
        except DataContractError:
            continue
    raise DataContractError("winwin dynamic parser failed for all date candidates")


def fetch_authoritative_latest_issue(timeout_s: float = 10.0) -> tuple[str, str]:
    try:
        page = httpx.get(WINWIN_URL, timeout=timeout_s)
        page.raise_for_status()
        dynamic_rows = _fetch_winwin_dynamic_records(page.text, timeout_s=timeout_s)
        if not dynamic_rows:
            raise DataContractError("authoritative dynamic endpoint returned no rows")
        return dynamic_rows[-1].issue, "winwin_dynamic"
    except Exception as exc:  # noqa: BLE001
        fallback = fetch_latest(sources=[WINWIN_URL, AUZO_URL], timeout_s=timeout_s)
        if not fallback.records:
            raise DataContractError("authoritative latest issue probe failed and fallback fetch returned empty") from exc
        latest_day = max(r.draw_date for r in fallback.records)
        same_day = sorted([r for r in fallback.records if r.draw_date == latest_day], key=lambda r: r.issue)
        if not same_day:
            raise DataContractError("authoritative latest issue probe fallback has no same-day rows") from exc
        return same_day[-1].issue, f"fallback:{fallback.source_url}"


def _parse_auzo_html(html: str) -> list[DrawRecord]:
    rows: list[dict[str, object]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)
        if len(cells) < 3:
            continue
        clean = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        issue_match = re.search(r"\d{8,}", clean[0])
        if not issue_match:
            continue
        issue = issue_match.group(0)
        draw_time = clean[1]
        nums: list[int] = []
        for c in clean[2:]:
            nums.extend(_numbers_from_text(c))
            if len(nums) >= 20:
                break
        if len(nums) != 20:
            continue
        rows.append({"issue": issue, "draw_date": draw_time, "numbers": nums})

    if not rows:
        regex_rows = []
        for m in re.finditer(r"(\d{8,})(.*?)(\d{1,2}(?:\D+\d{1,2}){19,40})", html, flags=re.DOTALL):
            issue = m.group(1)
            segment = m.group(2)
            nums = _numbers_from_text(m.group(3))[:20]
            time_match = re.search(r"\d{4}[\-/]\d{1,2}[\-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?", segment)
            draw_time = time_match.group(0) if time_match else datetime.now().date().isoformat()
            if len(nums) == 20:
                regex_rows.append({"issue": issue, "draw_date": draw_time, "numbers": nums})
        rows = regex_rows

    return _to_draw_records(rows, source_name="auzo")


def _check_source_health(html: str) -> None:
    lower = html.lower()
    for marker in _DB_ERROR_MARKERS:
        if marker in lower:
            raise DataContractError(f"source health check failed: found '{marker}'")


def _choose_source_parser(source: str):
    parsed = urlparse(source)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "winwin.tw" in host and path.startswith("/bingo"):
        return "winwin"
    if "lotto.auzo.tw" in host and path.endswith("/rk.php"):
        return "auzo"
    if "winwin.tw" in host:
        return "winwin"
    return "auzo"


def _parse_records_by_source(source: str, html: str, timeout_s: float) -> list[DrawRecord]:
    source_type = _choose_source_parser(source)
    if source_type == "winwin":
        try:
            return _parse_winwin_static_html(html)
        except DataContractError:
            if _has_dynamic_marker(html):
                return _fetch_winwin_dynamic_records(html, timeout_s=timeout_s)
            raise
    return _parse_auzo_html(html)


def parse_winwin_html(html: str) -> list[DrawRecord]:
    return _parse_winwin_static_html(html)


def fetch_latest(sources: list[str] | None = None, timeout_s: float = 10.0) -> FetchResult:
    srcs = sources or [WINWIN_URL, AUZO_URL]
    last_err = ""
    failover_reason: str | None = None
    for idx, source in enumerate(srcs, 1):
        response = None
        try:
            response = httpx.get(source, timeout=timeout_s)
            response.raise_for_status()
            _check_source_health(response.text)
            rows = _parse_records_by_source(source, response.text, timeout_s=timeout_s)
            if not rows:
                raise DataContractError("latest-day rows empty")
            if idx > 1 and failover_reason is None:
                failover_reason = "primary_source_failed_then_switched"
            return FetchResult(rows, source, datetime.now(timezone.utc).isoformat(timespec="seconds"), idx, failover_reason=failover_reason)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_err = f"transport error @ {source}: {exc}"
            _write_debug_snapshot("" if response is None else response.text, source, last_err)
            if idx == 1 and len(srcs) > 1:
                failover_reason = last_err
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = f"parse error @ {source}: {exc}"
            _write_debug_snapshot("" if response is None else response.text, source, last_err)
            if idx == 1 and len(srcs) > 1:
                failover_reason = last_err
            continue
    raise DataContractError(f"fetch failed for all sources: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/winwin_latest_processed.csv")
    parser.add_argument("--today-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gzip", action="store_true")
    parser.add_argument("--max-file-mb", type=float, default=95.0)
    args = parser.parse_args()

    result = fetch_latest()
    rows = result.records
    if args.today_only and rows:
        latest_day = max(r.draw_date for r in rows)
        rows = [r for r in rows if r.draw_date == latest_day]

    output_path = Path(args.output)
    if args.gzip or args.today_only:
        payload = [r.to_dict() for r in rows]
        write_json_gz_if_needed(
            payload,
            output_path.with_suffix(".json.gz"),
            max_file_mb=float(args.max_file_mb),
            producer_script="src.fetch_winwin",
        )
    write_processed(output_path, rows)
    enforce_file_size(output_path, max_bytes=int(args.max_file_mb * 1024 * 1024))


if __name__ == "__main__":
    main()
