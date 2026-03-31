from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from winwin_service.api import app

client = TestClient(app)


def test_health_smoke() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_integration_mocked_network(monkeypatch) -> None:
    from winwin_service import api
    api._PREDICTION_CACHE.clear()

    monkeypatch.setattr(
        api,
        'fetch_latest_draws',
        lambda *args, **kwargs: ([list(range(1, 21))] * 50, 114000123),
    )
    monkeypatch.setattr(
        api,
        'predict_top3',
        lambda draws, latest, config=None, include_regime_debug=False: {
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
    assert data['metadata']['cache_hit'] is False


def test_predict_fail_fast(monkeypatch) -> None:
    from winwin_service import api
    api._PREDICTION_CACHE.clear()

    monkeypatch.setattr(
        api,
        'fetch_latest_draws',
        lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(api.FetchError('upstream down')),
    )

    response = client.get('/predict')
    assert response.status_code == 502
    body = response.json()['detail']
    assert body['error_code'] == 'FETCH_FAILED'
    assert 'upstream down' in body['detail']


def test_predict_fail_fast_predict_error(monkeypatch) -> None:
    from winwin_service import api
    api._PREDICTION_CACHE.clear()

    monkeypatch.setattr(
        api,
        'fetch_latest_draws',
        lambda *args, **kwargs: ([list(range(1, 21))] * 20, 114000123),
    )
    monkeypatch.setattr(
        api,
        'predict_top3',
        lambda draws, latest, config=None, include_regime_debug=False: (
            _ for _ in ()
        ).throw(api.PredictError('no combinations')),
    )

    response = client.get('/predict')
    assert response.status_code == 502
    body = response.json()['detail']
    assert body['error_code'] == 'PREDICT_FAILED'
    assert 'no combinations' in body['detail']


def test_predict_cache_hit_without_recompute(monkeypatch) -> None:
    from winwin_service import api

    api._PREDICTION_CACHE.clear()
    calls = {"predict": 0, "fetch": 0}
    base_time = {"now": 1000.0}
    monkeypatch.setattr(api.time, 'time', lambda: base_time["now"])

    def _fake_fetch(*args, **kwargs):
        calls["fetch"] += 1
        return ([list(range(1, 21))] * 20, 114000200)

    monkeypatch.setattr(api, 'fetch_latest_draws', _fake_fetch)

    def _fake_predict(
        draws,
        latest,
        config=None,
        include_regime_debug=False,
    ):
        calls["predict"] += 1
        return {
            'target_period': latest + 1,
            'latest_period': latest,
            'top3': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'kill_zone': [10, 11],
            'metadata': {'analyzed_draws': len(draws)},
        }

    monkeypatch.setattr(api, 'predict_top3', _fake_predict)

    r1 = client.get('/predict')
    base_time["now"] += 5.0
    r2 = client.get('/predict')
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["predict"] == 1
    assert calls["fetch"] == 1
    assert r1.json()['metadata']['cache_hit'] is False
    assert r2.json()['metadata']['cache_hit'] is True
    assert r2.json()['metadata']['cache_strategy'] == 'ttl_before_fetch'


def test_predict_cache_expired_refetches(monkeypatch) -> None:
    from winwin_service import api

    api._PREDICTION_CACHE.clear()
    calls = {"predict": 0, "fetch": 0}
    base_time = {"now": 2000.0}
    monkeypatch.setattr(api.time, 'time', lambda: base_time["now"])

    def _fake_fetch(*args, **kwargs):
        calls["fetch"] += 1
        return ([list(range(1, 21))] * 20, 114000210)

    def _fake_predict(
        draws,
        latest,
        config=None,
        include_regime_debug=False,
    ):
        calls["predict"] += 1
        return {
            'target_period': latest + 1,
            'latest_period': latest,
            'top3': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'kill_zone': [10, 11],
            'metadata': {'analyzed_draws': len(draws)},
        }

    monkeypatch.setattr(api, 'fetch_latest_draws', _fake_fetch)
    monkeypatch.setattr(api, 'predict_top3', _fake_predict)

    r1 = client.get('/predict')
    base_time["now"] += 31.0
    r2 = client.get('/predict')
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["fetch"] == 2
    assert calls["predict"] == 2


def test_predict_debug_false_metadata_is_trimmed(monkeypatch) -> None:
    from winwin_service import api

    api._PREDICTION_CACHE.clear()
    monkeypatch.setattr(
        api,
        'fetch_latest_draws',
        lambda *args, **kwargs: ([list(range(1, 21))] * 20, 114000222),
    )
    monkeypatch.setattr(
        api,
        'predict_top3',
        lambda draws, latest, config=None, include_regime_debug=False: {
            'target_period': latest + 1,
            'latest_period': latest,
            'top3': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'kill_zone': [10, 11],
            'metadata': (
                {'regime_metrics_raw': {'k': 1}}
                if include_regime_debug
                else {'analyzed_draws': len(draws)}
            ),
        },
    )
    response = client.get('/predict')
    assert response.status_code == 200
    assert 'regime_metrics_raw' not in response.json()['metadata']


def test_load_validated_config_rejects_legacy_long_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from winwin_service import api

    payload = {
        "params": {
            "recent_draws_count": 50,
            "max_recent_draws_count": 50,
            "min_score_threshold": 60,
        }
    }
    cfg_path = tmp_path / "best_config.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(api, "_BEST_CONFIG_PATH", cfg_path)
    loaded, source, matched = api._load_validated_config()

    assert loaded.recent_draws_count <= 5
    assert loaded.max_recent_draws_count is not None
    assert loaded.max_recent_draws_count <= 5
    assert source == "default_config_reject_long_window_best_config"
    assert matched is False


def test_predict_passes_active_config_to_fetcher(monkeypatch) -> None:
    from winwin_service import api

    api._PREDICTION_CACHE.clear()
    captured = {}

    def _fake_fetch(*args, **kwargs):
        captured["config"] = kwargs.get("config")
        return ([list(range(1, 21))] * 12, 114000300)

    monkeypatch.setattr(api, "fetch_latest_draws", _fake_fetch)
    monkeypatch.setattr(
        api,
        "predict_top3",
        lambda draws, latest, config=None, include_regime_debug=False: {
            "target_period": latest + 1,
            "latest_period": latest,
            "top3": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "kill_zone": [10, 11],
            "metadata": {"analyzed_draws": len(draws)},
        },
    )

    response = client.get("/predict")
    assert response.status_code == 200
    assert captured["config"] == api._ACTIVE_CONFIG
