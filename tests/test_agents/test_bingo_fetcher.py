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
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])
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

    assert source == "https://www.pilio.idv.tw/bingo/list.asp"
    assert [record.issue for record in records] == [101, 102, 103]


def test_duplicate_issue_with_different_content_raises(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])
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
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])
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
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])
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
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])

    with pytest.raises(FetchDrawsError, match="DB Error"):
        fetcher._check_source_health("something DB Error happened")


def test_latest_issue_hint_mismatch_raises(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])
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
        sources=["https://www.pilio.idv.tw/bingo/list.asp"],
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
        fetcher._fetch_html("https://www.pilio.idv.tw/bingo/list.asp")

    assert calls["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_build_recent_draws_outputs_numbers(monkeypatch):
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])
    ordered = [
        DrawRecord(issue=101, draw_time="2026-01-01 00:00", numbers=_nums(1)),
        DrawRecord(issue=102, draw_time="2026-01-02 00:00", numbers=_nums(2)),
    ]

    monkeypatch.setattr(
        fetcher,
        "fetch_recent_records",
        lambda min_draws, max_draws: (
            ordered,
            "https://www.pilio.idv.tw/bingo/list.asp",
        ),
    )

    recent_draws, records, source = build_recent_draws(
        fetcher, min_draws=2, max_draws=50
    )

    assert source == "https://www.pilio.idv.tw/bingo/list.asp"
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


def test_parse_pilio_fixture_extracts_required_fields():
    fixture = """
    <tr style="text-align:center; background-color: #FFDBCE;"><td>
      <span>【期別: 115014398】</span><br />
      02,&nbsp;09,&nbsp;10,&nbsp;11,&nbsp;14,&nbsp;20,&nbsp;<span>22</span>,&nbsp;25,&nbsp;27,&nbsp;35,&nbsp;
      43,&nbsp;45,&nbsp;47,&nbsp;50,&nbsp;53,&nbsp;55,&nbsp;62,&nbsp;65,&nbsp;78,&nbsp;79<BR>
      <span>超級獎號:</span><span>22</span> _ <span>猜大小:</span><span>大</span> _ <span>猜單雙:</span><span>雙</span>
      <span>(22:40)</span>
    </td></tr>
    <tr style="text-align:center;"><td>
      <span>【期別: 115014399】</span><br />
      01,&nbsp;03,&nbsp;06,&nbsp;<span>07</span>,&nbsp;08,&nbsp;09,&nbsp;22,&nbsp;24,&nbsp;26,&nbsp;38,&nbsp;
      40,&nbsp;41,&nbsp;44,&nbsp;45,&nbsp;51,&nbsp;55,&nbsp;63,&nbsp;66,&nbsp;70,&nbsp;75<BR>
      <span>超級獎號:</span><span>07</span> _ <span>猜大小:</span><span>小</span> _ <span>猜單雙:</span><span>單</span>
      <span>(22:45)</span>
    </td></tr>
    """
    fetcher = BingoDrawFetcher(sources=["https://www.pilio.idv.tw/bingo/list.asp"])

    parsed = fetcher._parse_pilio_bingo_list(fixture)

    assert len(parsed) == 2
    normalized = [fetcher._normalize_row(row) for row in parsed]
    assert [r.issue for r in normalized] == [115014398, 115014399]
    assert [r.draw_time for r in normalized] == ["22:40", "22:45"]
    assert [r.super_number for r in normalized] == [22, 7]
    assert [r.big_small for r in normalized] == ["大", "小"]
    assert [r.odd_even for r in normalized] == ["雙", "單"]


def test_fallback_to_taiwan_lottery_when_pilio_fails(monkeypatch):
    fetcher = BingoDrawFetcher(
        sources=[
            "https://www.pilio.idv.tw/bingo/list.asp",
            "https://www.taiwanlottery.com.tw/lotto/bingobingo/history.aspx",
        ]
    )

    data_by_source = {
        "https://www.pilio.idv.tw/bingo/list.asp": "DB Error",
        "https://www.taiwanlottery.com.tw/lotto/bingobingo/history.aspx": "ok",
    }

    rows_by_source = {
        "https://www.taiwanlottery.com.tw/lotto/bingobingo/history.aspx": [
            {"issue": 201, "draw_time": "22:00", "numbers": _nums(1)},
            {"issue": 202, "draw_time": "22:05", "numbers": _nums(2)},
        ]
    }

    monkeypatch.setattr(fetcher, "_fetch_html", lambda source: data_by_source[source])
    monkeypatch.setattr(
        fetcher,
        "_parse_records_by_source",
        lambda source, _html: rows_by_source.get(source, []),
    )
    monkeypatch.setattr(
        fetcher,
        "_extract_latest_issue_hint_by_source",
        lambda source, _html: 202 if "taiwanlottery.com.tw" in source else None,
    )

    records, source = fetcher.fetch_recent_records(min_draws=2, max_draws=10)

    assert source == "https://www.taiwanlottery.com.tw/lotto/bingobingo/history.aspx"
    assert [r.issue for r in records] == [201, 202]


def test_parse_winwin_fixture_extracts_required_fields():
    fixture = """
    <table>
      <tr><th>期別/時間</th><th>號碼</th><th>連莊球數</th><th>大小</th><th>單雙</th></tr>
      <tr>
        <td>115014500 22:50</td>
        <td>01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20</td>
        <td>連莊球數 2</td>
        <td>大</td>
        <td>單</td>
      </tr>
      <tr>
        <td>115014501 22:55</td>
        <td>21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40</td>
        <td>連莊球數 1</td>
        <td>小</td>
        <td>雙</td>
      </tr>
    </table>
    """
    fetcher = BingoDrawFetcher(sources=["https://winwin.tw/Bingo"])

    parsed = fetcher._parse_winwin_bingo(fixture)
    normalized = [fetcher._normalize_row(row) for row in parsed]

    assert [r.issue for r in normalized] == [115014500, 115014501]
    assert [r.draw_time for r in normalized] == ["22:50", "22:55"]
    assert all(len(r.numbers) == 20 for r in normalized)
    assert [r.size_label for r in normalized] == ["大", "小"]
    assert [r.odd_even_label for r in normalized] == ["單", "雙"]
    assert [r.streak_count for r in normalized] == [2, 1]
