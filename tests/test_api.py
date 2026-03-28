from __future__ import annotations

from fastapi.testclient import TestClient

from winwin_service.api import app

client = TestClient(app)


def test_health_smoke() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_integration_mocked_network(monkeypatch) -> None:
    from winwin_service import api

    monkeypatch.setattr(
        api,
        'fetch_latest_draws',
        lambda: ([list(range(1, 21))] * 50, 114000123),
    )
    monkeypatch.setattr(
        api,
        'predict_top3',
        lambda draws, latest: {
            'target_period': latest + 1,
            'latest_period': latest,
            'top3': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'kill_zone': [10, 11],
            'metadata': {'analyzed_draws': len(draws)},
        },
    )

    response = client.get('/predict')
    assert response.status_code == 200
    data = response.json()
    assert data['target_period'] == 114000124
    assert len(data['top3']) == 3


def test_predict_fail_fast(monkeypatch) -> None:
    from winwin_service import api

    monkeypatch.setattr(
        api,
        'fetch_latest_draws',
        lambda: (_ for _ in ()).throw(api.FetchError('upstream down')),
    )

    response = client.get('/predict')
    assert response.status_code == 502
    assert 'upstream down' in response.json()['detail']
