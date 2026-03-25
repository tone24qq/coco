import json

import pytest

from src.fetch_latest import DEFAULT_HEADERS, FetchConfig, fetch_latest


class DummyResponse:
    def __init__(self, status_code: int, text: str = "", payload=None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def get(self, url: str, timeout: float, headers=None):
        self.calls.append({"url": url, "timeout": timeout, "headers": headers or {}})
        key = (url, timeout)
        if key not in self.mapping and url.startswith(
            "https://winwin.tw/Bingo/GetBingoData?date="
        ):
            key = ("WINWIN_JSON", timeout)
        if key not in self.mapping:
            raise RuntimeError(f"unexpected request: {key}")
        return self.mapping[key]


def _nums() -> str:
    return "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"


def _nums_alt() -> str:
    return "21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"


def test_winwin_json_new_fields_parse_success() -> None:
    client = DummyClient(
        {
            ("WINWIN_JSON", 1.0): DummyResponse(
                status_code=200,
                payload=[
                    {
                        "Issue": "115000001",
                        "DrawTime": "2026-03-24 12:00",
                        "BingoNumbers": list(range(1, 21)),
                    },
                    {
                        "Issue": "115000002",
                        "DrawTime": "2026-03-24 12:05",
                        "BingoNumbers": list(range(1, 21)),
                    },
                    {
                        "Issue": "115000003",
                        "DrawTime": "2026-03-24 12:10",
                        "BingoNumbers": list(range(1, 21)),
                    },
                    {
                        "Issue": "115000004",
                        "DrawTime": "2026-03-24 12:15",
                        "BingoNumbers": list(range(1, 21)),
                    },
                    {
                        "Issue": "115000005",
                        "DrawTime": "2026-03-24 12:20",
                        "BingoNumbers": list(range(1, 21)),
                    },
                ],
            )
        }
    )

    records, _, attempts, diagnostics = fetch_latest(
        sources=[{"name": "winwin", "url": "https://winwin.tw/Bingo"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if len(records) != 5 or attempts[-1].get("parser_path") != "json":
        pytest.fail("winwin new json format should parse")
    if diagnostics["max_observed_issue"] != "115000005":
        pytest.fail("diagnostics should include max observed issue")
    if not client.calls or "User-Agent" not in client.calls[0]["headers"]:
        pytest.fail("browser-like headers are required")
    if client.calls[0]["headers"].get("Accept-Language") != DEFAULT_HEADERS.get(
        "Accept-Language"
    ):
        pytest.fail("browser-like headers are required")


def test_source_specific_fail_then_generic_fallback() -> None:
    good = (
        f"115000001 {_nums()} "
        f"115000002 {_nums()} "
        f"115000003 {_nums()} "
        f"115000004 {_nums()} "
        f"115000005 {_nums()}"
    )
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=good)}
    )

    records, _, attempts, _ = fetch_latest(
        sources=[{"name": "unknown", "url": "https://x"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if len(records) != 5 or attempts[-1].get("parser_path") != "generic":
        pytest.fail("generic parser should be fallback path")


def test_all_sources_failed_error_contains_attempt_details() -> None:
    client = DummyClient(
        {
            ("https://a", 1.0): DummyResponse(status_code=500, text="err"),
            ("https://b", 1.0): DummyResponse(status_code=500, text="err"),
        }
    )
    with pytest.raises(RuntimeError) as exc_info:
        fetch_latest(
            sources=[
                {"name": "pilio", "url": "https://a"},
                {"name": "pilio", "url": "https://b"},
            ],
            config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
            client=client,
        )

    message = str(exc_info.value)
    if "attempts=" not in message or "parser_path" not in message:
        pytest.fail("error should contain attempts details")
    payload = message.split("attempts=", 1)[1]
    parsed = json.loads(payload)
    if not parsed or parsed[0]["status"] != "error":
        pytest.fail("attempt details should be structured and include error status")


def test_winwin_old_but_auzo_newer_selects_auzo() -> None:
    winwin_html = (
        f"115016845 {_nums()} "
        f"115016846 {_nums()} "
        f"115016847 {_nums()} "
        f"115016848 {_nums()} "
        f"115016849 {_nums()}"
    )
    auzo_html = (
        f"115016856 {_nums()} "
        f"115016857 {_nums()} "
        f"115016858 {_nums()} "
        f"115016859 {_nums()} "
        f"115016860 {_nums()}"
    )
    client = DummyClient(
        {
            ("WINWIN_JSON", 1.0): DummyResponse(status_code=500),
            ("https://winwin.tw/Bingo", 1.0): DummyResponse(
                status_code=200, text=winwin_html
            ),
            ("https://www.auzo.tw/RJ.php", 1.0): DummyResponse(
                status_code=200, text=auzo_html
            ),
        }
    )

    records, data_source, attempts, diagnostics = fetch_latest(
        sources=[
            {"name": "winwin", "url": "https://winwin.tw/Bingo"},
            {"name": "auzo", "url": "https://www.auzo.tw/RJ.php"},
        ],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )

    if data_source != "https://www.auzo.tw/RJ.php":
        pytest.fail("newer source should be selected")
    if records[-1]["issue"] != "115016860":
        pytest.fail("selected tail should match newer latest issue")
    if not any(item.get("source") == "winwin" for item in attempts):
        pytest.fail("must evaluate all configured sources")
    if diagnostics["source_latest_issues"] != {
        "winwin": "115016849",
        "auzo": "115016860",
    }:
        pytest.fail("source latest issues diagnostics mismatch")


def test_same_latest_issue_uses_longer_tail() -> None:
    short_tail = (
        f"115016856 {_nums()} "
        f"115016857 {_nums()} "
        f"115016858 {_nums()} "
        f"115016859 {_nums()} "
        f"115016860 {_nums()}"
    )
    long_tail = (
        f"115016853 {_nums()} "
        f"115016854 {_nums()} "
        f"115016855 {_nums()} "
        f"115016856 {_nums()} "
        f"115016857 {_nums()} "
        f"115016858 {_nums()} "
        f"115016859 {_nums()} "
        f"115016860 {_nums()}"
    )
    client = DummyClient(
        {
            ("https://winwin.tw/Bingo", 1.0): DummyResponse(
                status_code=200, text=short_tail
            ),
            ("WINWIN_JSON", 1.0): DummyResponse(status_code=500),
            ("https://www.auzo.tw/RJ.php", 1.0): DummyResponse(
                status_code=200, text=long_tail
            ),
        }
    )

    _, data_source, _, diagnostics = fetch_latest(
        sources=[
            {"name": "winwin", "url": "https://winwin.tw/Bingo"},
            {"name": "auzo", "url": "https://www.auzo.tw/RJ.php"},
        ],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )

    if data_source != "https://www.auzo.tw/RJ.php":
        pytest.fail("same issue should use longer tail source")
    if "longest records tail" not in diagnostics["selected_source_reason"]:
        pytest.fail("selection reason should include tie-break detail")


def test_selected_source_returns_full_records_not_tail() -> None:
    html = (
        f"115016801 {_nums()} "
        f"115016802 {_nums()} "
        f"115016804 {_nums()} "
        f"115016805 {_nums()} "
        f"115016806 {_nums()} "
        f"115016807 {_nums()} "
        f"115016808 {_nums()} "
        f"115016809 {_nums()} "
        f"115016810 {_nums()}"
    )
    client = DummyClient(
        {("https://www.auzo.tw/RJ.php", 1.0): DummyResponse(status_code=200, text=html)}
    )

    records, _, _, diagnostics = fetch_latest(
        sources=[{"name": "auzo", "url": "https://www.auzo.tw/RJ.php"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )

    issues = [int(item["issue"]) for item in records]
    if issues != [
        115016801,
        115016802,
        115016804,
        115016805,
        115016806,
        115016807,
        115016808,
        115016809,
        115016810,
    ]:
        pytest.fail("fetch_latest should return selected source full_records")
    if diagnostics["selected_source_full_records_count"] != 9:
        pytest.fail("diagnostics should expose selected full records count")
    if diagnostics["selected_source_tail_count"] != 7:
        pytest.fail("diagnostics should expose selected tail count")


def test_parser_fail_does_not_block_other_sources() -> None:
    bad = "only garbage"
    good = (
        f"115016856 {_nums()} "
        f"115016857 {_nums()} "
        f"115016858 {_nums()} "
        f"115016859 {_nums()} "
        f"115016860 {_nums()}"
    )
    client = DummyClient(
        {
            ("https://www.auzo.tw/RJ.php", 1.0): DummyResponse(
                status_code=200, text=bad
            ),
            ("https://www.pilio.idv.tw/bingo/list.asp", 1.0): DummyResponse(
                status_code=200, text=good
            ),
        }
    )

    records, data_source, attempts, _ = fetch_latest(
        sources=[
            {"name": "auzo", "url": "https://www.auzo.tw/RJ.php"},
            {"name": "pilio", "url": "https://www.pilio.idv.tw/bingo/list.asp"},
        ],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )

    if (
        data_source != "https://www.pilio.idv.tw/bingo/list.asp"
        or records[-1]["issue"] != "115016860"
    ):
        pytest.fail("failed parser source should not block usable source")
    if not any(
        item["source"] == "auzo" and item["status"] == "error" for item in attempts
    ):
        pytest.fail("parser failure attempt should be recorded")


def test_same_issue_numbers_conflict_marked_divergent() -> None:
    winwin_html = (
        f"115016856 {_nums()} "
        f"115016857 {_nums()} "
        f"115016858 {_nums()} "
        f"115016859 {_nums()} "
        f"115016860 {_nums()}"
    )
    auzo_html = (
        f"115016856 {_nums()} "
        f"115016857 {_nums()} "
        f"115016858 {_nums()} "
        f"115016859 {_nums()} "
        f"115016860 {_nums_alt()}"
    )
    client = DummyClient(
        {
            ("WINWIN_JSON", 1.0): DummyResponse(status_code=500),
            ("https://winwin.tw/Bingo", 1.0): DummyResponse(
                status_code=200, text=winwin_html
            ),
            ("https://www.auzo.tw/RJ.php", 1.0): DummyResponse(
                status_code=200, text=auzo_html
            ),
        }
    )

    _, _, _, diagnostics = fetch_latest(
        sources=[
            {"name": "winwin", "url": "https://winwin.tw/Bingo"},
            {"name": "auzo", "url": "https://www.auzo.tw/RJ.php"},
        ],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )

    if diagnostics["consensus_status"] != "divergent":
        pytest.fail("numbers conflict must be divergent")
    conflicts = diagnostics["source_consensus"]["conflicts"]
    if not conflicts or conflicts[0]["issue"] != "115016860":
        pytest.fail("conflict diagnostics should include conflicted issue")


def test_source_records_count_and_tail_count_diagnostics() -> None:
    html = (
        f"115016801 {_nums()} "
        f"115016802 {_nums()} "
        f"115016803 {_nums()} "
        f"115016804 {_nums()} "
        f"115016805 {_nums()} "
        f"115016806 {_nums()} "
    )
    client = DummyClient(
        {("https://www.auzo.tw/RJ.php", 1.0): DummyResponse(status_code=200, text=html)}
    )
    _, _, _, diagnostics = fetch_latest(
        sources=[{"name": "auzo", "url": "https://www.auzo.tw/RJ.php"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if diagnostics["source_records_count"] != {"auzo": 6}:
        pytest.fail("source_records_count should use full records count")
    if diagnostics["source_tail_count"] != {"auzo": 6}:
        pytest.fail("source_tail_count should be exposed")
