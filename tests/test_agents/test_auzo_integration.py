from __future__ import annotations

from urllib import error

from src.integrations.auzo import AuzoConfig, fetch_auzo_external_analysis


def test_auzo_fetch_degraded_when_network_error(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise error.URLError("boom")

    monkeypatch.setattr("src.integrations.auzo._http_get", _raise)
    payload = fetch_auzo_external_analysis(
        AuzoConfig(timeout_seconds=0.1, ttl_seconds=0)
    )
    assert payload["external_status"] == "degraded"
    assert payload["provider"] == "auzo"
    assert payload["external_analysis"] == {}
