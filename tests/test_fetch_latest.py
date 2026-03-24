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
    def __init__(self, responses):
        self.responses = responses

    def get(self, url: str, timeout: float):
        key = (url, timeout)
        if key not in self.responses:
            raise RuntimeError(f"unexpected request: {key}")
        return self.responses[key]


def test_fetch_latest_success() -> None:
    html = "115000001 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    client = DummyClient(
        {
            ("https://example.com", 3.0): DummyResponse(status_code=200, text=html),
        }
    )
    records, source, attempts = fetch_latest(
        sources=[{"name": "mock", "url": "https://example.com"}],
        config=FetchConfig(timeout_seconds=3.0, retries=0, backoff_seconds=0.0),
        client=client,
    )
    if not records or source != "https://example.com" or not attempts:
        pytest.fail("fetch_latest should return records/source/attempts")


def test_fetch_latest_failure_clear_error() -> None:
    client = DummyClient(
        {
            ("https://example.com", 3.0): DummyResponse(status_code=500, text="fail"),
        }
    )
    with pytest.raises(RuntimeError, match="All sources failed"):
        fetch_latest(
            sources=[{"name": "mock", "url": "https://example.com"}],
            config=FetchConfig(timeout_seconds=3.0, retries=0, backoff_seconds=0.0),
            client=client,
        )
