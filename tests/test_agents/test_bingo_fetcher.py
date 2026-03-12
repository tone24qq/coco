from pathlib import Path

import pytest

from src.fetchers.auzo_bingo import (
    BingoDrawFetcher,
    DrawRecord,
    FetchDrawsError,
    build_recent_draws,
)


def _nums(start: int) -> list[int]:
    return [((start + i - 1) % 80) + 1 for i in range(1, 21)]


def test_reorders_new_to_old_into_old_to_new(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])
    rows = [
        {"issue": 103, "draw_time": "2026-01-03 00:00", "numbers": _nums(3)},
        {"issue": 102, "draw_time": "2026-01-02 00:00", "numbers": _nums(2)},
        {"issue": 101, "draw_time": "2026-01-01 00:00", "numbers": _nums(1)},
    ]

    monkeypatch.setattr(fetcher, "_fetch_html", lambda _source: "ok")
    monkeypatch.setattr(fetcher, "_check_source_health", lambda _html: None)
    monkeypatch.setattr(
        fetcher, "_parse_records_by_source", lambda _source, _html: rows
    )
    monkeypatch.setattr(
        fetcher, "_extract_latest_issue_hint_by_source", lambda _source, _html: 103
    )

    records, source = fetcher.fetch_recent_records(min_draws=3, max_draws=50)

    assert source == "https://lotto.auzo.tw/bingobingoV1.php"
    assert [record.issue for record in records] == [101, 102, 103]


def test_duplicate_issue_with_different_content_raises(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])
    rows = [
        {"issue": 101, "draw_time": "2026-01-01 00:00", "numbers": _nums(1)},
        {"issue": 101, "draw_time": "2026-01-01 00:00", "numbers": _nums(2)},
        {"issue": 102, "draw_time": "2026-01-02 00:00", "numbers": _nums(3)},
    ]

    monkeypatch.setattr(fetcher, "_fetch_html", lambda _source: "ok")
    monkeypatch.setattr(fetcher, "_check_source_health", lambda _html: None)
    monkeypatch.setattr(
        fetcher, "_parse_records_by_source", lambda _source, _html: rows
    )
    monkeypatch.setattr(
        fetcher, "_extract_latest_issue_hint_by_source", lambda _source, _html: 102
    )

    with pytest.raises(FetchDrawsError, match="duplicate issue"):
        fetcher.fetch_recent_records(min_draws=2, max_draws=50)


def test_non_consecutive_issues_raises(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])
    rows = [
        {"issue": 101, "draw_time": "2026-01-01 00:00", "numbers": _nums(1)},
        {"issue": 103, "draw_time": "2026-01-03 00:00", "numbers": _nums(3)},
    ]

    monkeypatch.setattr(fetcher, "_fetch_html", lambda _source: "ok")
    monkeypatch.setattr(fetcher, "_check_source_health", lambda _html: None)
    monkeypatch.setattr(
        fetcher, "_parse_records_by_source", lambda _source, _html: rows
    )
    monkeypatch.setattr(
        fetcher, "_extract_latest_issue_hint_by_source", lambda _source, _html: 103
    )

    with pytest.raises(FetchDrawsError, match="not consecutive"):
        fetcher.fetch_recent_records(min_draws=2, max_draws=50)


def test_non_20_numbers_raises(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])
    rows = [
        {"issue": 101, "draw_time": "2026-01-01 00:00", "numbers": list(range(1, 20))},
        {"issue": 102, "draw_time": "2026-01-02 00:00", "numbers": _nums(2)},
    ]

    monkeypatch.setattr(fetcher, "_fetch_html", lambda _source: "ok")
    monkeypatch.setattr(fetcher, "_check_source_health", lambda _html: None)
    monkeypatch.setattr(
        fetcher, "_parse_records_by_source", lambda _source, _html: rows
    )
    monkeypatch.setattr(
        fetcher, "_extract_latest_issue_hint_by_source", lambda _source, _html: 102
    )

    with pytest.raises(FetchDrawsError, match="exactly 20"):
        fetcher.fetch_recent_records(min_draws=2, max_draws=50)


def test_source_healthcheck_db_error_raises():
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])

    with pytest.raises(FetchDrawsError, match="DB Error"):
        fetcher._check_source_health("something DB Error happened")


def test_latest_issue_hint_mismatch_raises(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])
    rows = [
        {"issue": 101, "draw_time": "2026-01-01 00:00", "numbers": _nums(1)},
        {"issue": 102, "draw_time": "2026-01-02 00:00", "numbers": _nums(2)},
    ]

    monkeypatch.setattr(fetcher, "_fetch_html", lambda _source: "ok")
    monkeypatch.setattr(fetcher, "_check_source_health", lambda _html: None)
    monkeypatch.setattr(
        fetcher, "_parse_records_by_source", lambda _source, _html: rows
    )
    monkeypatch.setattr(
        fetcher, "_extract_latest_issue_hint_by_source", lambda _source, _html: 999
    )

    with pytest.raises(FetchDrawsError, match="latest issue mismatch"):
        fetcher.fetch_recent_records(min_draws=2, max_draws=50)


def test_retry_uses_exponential_backoff(monkeypatch):
    fetcher = BingoDrawFetcher(
        sources=["https://lotto.auzo.tw/bingobingoV1.php"],
        retries=2,
        retry_backoff_seconds=0.1,
    )
    sleeps: list[float] = []

    monkeypatch.setattr(
        "src.fetchers.auzo_bingo.time.sleep", lambda x: sleeps.append(x)
    )

    calls = {"count": 0}

    def _raise_then_fail(_source):
        calls["count"] += 1
        raise OSError("boom")

    monkeypatch.setattr(
        "src.fetchers.auzo_bingo.request.urlopen",
        lambda source, timeout: _raise_then_fail(source),
    )

    with pytest.raises(FetchDrawsError, match="fetch failed"):
        fetcher._fetch_html("https://lotto.auzo.tw/bingobingoV1.php")

    assert calls["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_build_recent_draws_outputs_numbers(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])
    ordered = [
        DrawRecord(issue=101, draw_time="2026-01-01 00:00", numbers=_nums(1)),
        DrawRecord(issue=102, draw_time="2026-01-02 00:00", numbers=_nums(2)),
    ]

    monkeypatch.setattr(
        fetcher,
        "fetch_recent_records",
        lambda min_draws, max_draws: (
            ordered,
            "https://lotto.auzo.tw/bingobingoV1.php",
        ),
    )

    recent_draws, records, source = build_recent_draws(
        fetcher, min_draws=2, max_draws=50
    )

    assert source == "https://lotto.auzo.tw/bingobingoV1.php"
    assert len(recent_draws) == len(records) == 2
    assert recent_draws[0] == _nums(1)


def test_parse_bingobingov1_fixture_extracts_consecutive_issues_and_20_numbers(
    monkeypatch,
):
    fixture = Path("tests/fixtures/bingobingoV1_fragment.html").read_text(
        encoding="utf-8"
    )
    fetcher = BingoDrawFetcher(sources=["https://lotto.auzo.tw/bingobingoV1.php"])

    monkeypatch.setattr(fetcher, "_fetch_html", lambda _source: fixture)

    records, source = fetcher.fetch_recent_records(min_draws=3, max_draws=50)

    assert source == "https://lotto.auzo.tw/bingobingoV1.php"
    assert [record.issue for record in records] == [115014377, 115014378, 115014379]
    assert [record.draw_time for record in records] == ["20:55", "21:00", "21:05"]
    assert all(len(record.numbers) == 20 for record in records)
