import pytest

from src.fetch_latest import FetchConfig, fetch_latest


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

    def get(self, url: str, timeout: float):
        key = (url, timeout)
        if key not in self.mapping:
            raise RuntimeError(f"unexpected request: {key}")
        return self.mapping[key]


def test_source_error_in_attempts() -> None:
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=500, text="err")}
    )
    with pytest.raises(RuntimeError, match="All sources failed"):
        fetch_latest(
            sources=[{"name": "pilio", "url": "https://x"}],
            config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
            client=client,
        )


def test_source_specific_parser_success() -> None:
    html = "115000001 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=html)}
    )
    records, source, attempts = fetch_latest(
        sources=[{"name": "pilio", "url": "https://x"}],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if not records or source != "https://x" or attempts[-1]["status"] != "ok":
        pytest.fail("fetch_latest source parser failed")


def test_latest_issue_hint_mismatch_fail() -> None:
    html = "115000009 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 " "115000010"
    client = DummyClient(
        {("https://x", 1.0): DummyResponse(status_code=200, text=html)}
    )
    with pytest.raises(RuntimeError, match="All sources failed"):
        fetch_latest(
            sources=[{"name": "pilio", "url": "https://x"}],
            config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
            client=client,
        )


def test_source_failover_success() -> None:
    bad = DummyResponse(status_code=500, text="err")
    good_html = "115000001 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    client = DummyClient(
        {
            ("https://bad", 1.0): bad,
            ("https://good", 1.0): DummyResponse(status_code=200, text=good_html),
        }
    )
    records, source, attempts = fetch_latest(
        sources=[
            {"name": "pilio", "url": "https://bad"},
            {"name": "pilio", "url": "https://good"},
        ],
        config=FetchConfig(timeout_seconds=1.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if source != "https://good" or len(records) != 1:
        pytest.fail("source failover did not reach healthy source")
    if not any(item.get("status") == "error" for item in attempts):
        pytest.fail("fetch attempts should record source errors")
