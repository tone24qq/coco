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

    records, _, attempts = fetch_latest(
        sources=[{"name": "winwin", "url": "https://winwin.tw/Bingo"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if len(records) != 5 or attempts[-1].get("parser_path") != "json":
        pytest.fail("winwin new json format should parse")
    if not client.calls or "User-Agent" not in client.calls[0]["headers"]:
        pytest.fail("browser-like headers are required")
    if client.calls[0]["headers"].get("Accept-Language") != DEFAULT_HEADERS.get(
        "Accept-Language"
    ):
        pytest.fail("browser-like headers are required")


def test_winwin_json_old_fields_parse_success() -> None:
    nums_csv = _nums()
    client = DummyClient(
        {
            ("WINWIN_JSON", 1.0): DummyResponse(
                status_code=200,
                payload=[
                    {
                        "No": "115000002",
                        "OpenDate": "2026-03-24 12:05",
                        "BigShowOrder": nums_csv,
                    },
                    {
                        "No": "115000003",
                        "OpenDate": "2026-03-24 12:10",
                        "BigShowOrder": nums_csv,
                    },
                    {
                        "No": "115000004",
                        "OpenDate": "2026-03-24 12:15",
                        "BigShowOrder": nums_csv,
                    },
                    {
                        "No": "115000005",
                        "OpenDate": "2026-03-24 12:20",
                        "BigShowOrder": nums_csv,
                    },
                    {
                        "No": "115000006",
                        "OpenDate": "2026-03-24 12:25",
                        "BigShowOrder": nums_csv,
                    },
                ],
            )
        }
    )

    records, _, _ = fetch_latest(
        sources=[{"name": "winwin", "url": "https://winwin.tw/Bingo"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if records[0]["issue"] != "115000002":
        pytest.fail("winwin old json format should parse")


def test_big_show_order_string_split_to_20_unique_numbers() -> None:
    nums_spaced = "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"
    client = DummyClient(
        {
            ("WINWIN_JSON", 1.0): DummyResponse(
                status_code=200,
                payload=[
                    {
                        "No": "115000003",
                        "OpenDate": "2026-03-24 12:10",
                        "BigShowOrder": nums_spaced,
                    },
                    {
                        "No": "115000004",
                        "OpenDate": "2026-03-24 12:15",
                        "BigShowOrder": nums_spaced,
                    },
                    {
                        "No": "115000005",
                        "OpenDate": "2026-03-24 12:20",
                        "BigShowOrder": nums_spaced,
                    },
                    {
                        "No": "115000006",
                        "OpenDate": "2026-03-24 12:25",
                        "BigShowOrder": nums_spaced,
                    },
                    {
                        "No": "115000007",
                        "OpenDate": "2026-03-24 12:30",
                        "BigShowOrder": nums_spaced,
                    },
                ],
            )
        }
    )

    records, _, _ = fetch_latest(
        sources=[{"name": "winwin", "url": "https://winwin.tw/Bingo"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    numbers = records[0]["numbers"]
    if len(numbers) != 20 or len(set(numbers)) != 20:
        pytest.fail("BigShowOrder string must split into 20 unique numbers")


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

    records, _, attempts = fetch_latest(
        sources=[{"name": "unknown", "url": "https://x"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if len(records) != 5 or attempts[-1].get("parser_path") != "generic":
        pytest.fail("generic parser should be fallback path")


def test_hint_mismatch_but_records_reasonable_should_not_fail() -> None:
    html = (
        f"115000006 {_nums()} "
        f"115000007 {_nums()} "
        f"115000008 {_nums()} "
        f"115000009 {_nums()} "
        f"115000010 {_nums()} "
        "115000090"
    )
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=html)}
    )

    records, _, _ = fetch_latest(
        sources=[{"name": "pilio", "url": "https://x"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if records[-1]["issue"] != "115000010":
        pytest.fail("reasonable hint mismatch should not hard fail")


def test_tail_consecutive_accepts_gap_in_older_history() -> None:
    html = (
        f"115000001 {_nums()} "
        f"115000002 {_nums()} "
        f"115000004 {_nums()} "
        f"115000005 {_nums()} "
        f"115000006 {_nums()} "
        f"115000007 {_nums()} "
        f"115000008 {_nums()}"
    )
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=html)}
    )

    records, _, _ = fetch_latest(
        sources=[{"name": "pilio", "url": "https://x"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if records[0]["issue"] != "115000004" or records[-1]["issue"] != "115000008":
        pytest.fail("should keep newest consecutive tail")


def test_benign_duplicate_issue_deduped() -> None:
    html = (
        f"115000010 {_nums()} "
        f"115000010 {_nums()} "
        f"115000011 {_nums()} "
        f"115000012 {_nums()} "
        f"115000013 {_nums()} "
        f"115000014 {_nums()}"
    )
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=html)}
    )

    records, _, _ = fetch_latest(
        sources=[{"name": "pilio", "url": "https://x"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    issues = [r["issue"] for r in records]
    if issues.count("115000010") != 1:
        pytest.fail("benign duplicate issue should be deduped")


def test_conflicting_duplicate_issue_fail_fast() -> None:
    html = (
        "115000010 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 "
        "115000010 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21"
    )
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=html)}
    )

    with pytest.raises(RuntimeError, match="conflicting duplicate"):
        fetch_latest(
            sources=[{"name": "pilio", "url": "https://x"}],
            config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
            client=client,
        )


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
