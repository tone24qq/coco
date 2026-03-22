import threading
from dataclasses import replace
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import Mock

import src.api as api_module
from src.api import app
from src.predict import _load_recent_draws
from src.utils import log_progress


class _FakeRuntimeState:
    retrieval_index_version = "test"

    class _Recent:
        cache_status = "hit"
        updated_at_epoch = 0.0

    recent_cache = _Recent()


def test_health_and_predict_schema_with_mocked_runtime(monkeypatch, synthetic_records) -> None:
    class FakeArtifacts:
        feature_columns = ["cand_hits_last_100"]
        metadata = {"created_at": "2026-01-01T00:00:00", "model_family": "test"}

    captured: dict[str, str] = {}

    def fake_run_prediction(_artifacts, _cfg, _recent, request_id=None, response_mode="full", runtime_state=None):
        captured["response_mode"] = response_mode
        return {
            "issue": "20260101099",
            "source": "manual",
            "dynamic_context_n": 30,
            "top20_numbers": list(range(1, 21)),
            "big_count": 0,
            "small_count": 20,
            "odd_count": 10,
            "even_count": 10,
            "metadata": {
                "latest_fetched_issue": "20260101098",
                "fetched_same_day_issue_min": "20260101070",
                "fetched_same_day_issue_max": "20260101098",
                "fetched_same_day_issue_count": 29,
                "dynamic_context_n": 30,
                "runtime_history_issue_range": ["20260101098", "20260101098"],
            },
        }

    monkeypatch.setattr(
        api_module,
        "get_runtime",
        lambda: (
            FakeArtifacts(),
            {"history": {"min_dynamic_n": 20, "processed_path": "data/processed/history_processed.csv", "runtime_artifact_dir": "data/runtime_history"}, "auto_fetch": {"source": "x"}},
            None,
        ),
    )
    monkeypatch.setattr(api_module, "build_prediction_runtime_state", lambda artifacts, cfg: _FakeRuntimeState())
    monkeypatch.setattr(api_module, "run_prediction", fake_run_prediction)
    app.state.runtime_state = _FakeRuntimeState()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert "processed_history_exists" in health.json()
    assert "compact_history_ready" in health.json()

    payload = {"recent_draws": [r.to_dict() for r in synthetic_records[-30:]]}
    pred = client.post("/predict", json=payload)
    assert pred.status_code == 200
    body = pred.json()
    assert sorted(body.keys()) == sorted(
        [
            "latest_fetched_issue",
            "fetched_same_day_issue_min",
            "fetched_same_day_issue_max",
            "fetched_same_day_issue_count",
            "dynamic_context_n",
            "target_issue",
            "top20_numbers",
            "big_count",
            "small_count",
            "odd_count",
            "even_count",
            "size_summary",
            "odd_even_summary",
        ]
    )
    assert len(body["top20_numbers"]) == 20
    assert body["big_count"] + body["small_count"] == 20
    assert body["odd_count"] + body["even_count"] == 20
    assert body["fetched_same_day_issue_min"] == "20260101070"
    assert body["fetched_same_day_issue_max"] == "20260101098"
    assert body["fetched_same_day_issue_count"] == 29
    assert body["dynamic_context_n"] == 30
    assert "ranking_score_table" not in body
    assert "retrieval_top_matches" not in body
    assert captured["response_mode"] == "minimal"


def test_health_degraded_when_artifacts_missing(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_runtime",
        lambda: (
            None,
            {"history": {"min_dynamic_n": 20, "processed_path": "missing.csv", "runtime_artifact_dir": "data/runtime_history"}, "auto_fetch": {"sources": ["s"]}},
            "missing",
        ),
    )
    monkeypatch.setattr(api_module, "build_prediction_runtime_state", lambda artifacts, cfg: _FakeRuntimeState())
    app.state.runtime_state = _FakeRuntimeState()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_get_runtime_normalizes_config_paths_to_absolute(monkeypatch, tmp_path):
    class FakeArtifacts:
        feature_columns = ["x"]
        metadata = {"model_version": "test"}

    monkeypatch.chdir(tmp_path)
    api_module.get_runtime.cache_clear()
    monkeypatch.setattr(api_module, "load_artifacts", lambda _path: FakeArtifacts())
    monkeypatch.setattr(api_module, "build_prediction_runtime_state", lambda artifacts, cfg: object())

    artifacts, cfg, err = api_module.get_runtime()
    assert err is None
    assert artifacts is not None
    assert Path(cfg["models"]["dir"]).is_absolute()
    assert Path(cfg["history"]["processed_path"]).is_absolute()
    assert Path(cfg["provenance"]["audit_path"]).is_absolute()
    assert Path(cfg["snapshot"]["path"]).is_absolute()


def test_predict_singleflight_rejects_concurrent_request(monkeypatch, synthetic_records) -> None:
    class FakeArtifacts:
        feature_columns = ["cand_hits_last_100"]
        metadata = {"created_at": "2026-01-01T00:00:00", "model_family": "test"}

    lock_entered = threading.Event()
    release_lock = threading.Event()

    def fake_run_prediction(_artifacts, _cfg, _recent, request_id=None, response_mode="full", runtime_state=None):
        lock_entered.set()
        release_lock.wait(timeout=1.0)
        return {
            "issue": "20260101099",
            "top20_numbers": list(range(1, 21)),
            "big_count": 0,
            "small_count": 20,
            "odd_count": 10,
            "even_count": 10,
            "metadata": {
                "latest_fetched_issue": "20260101098",
                "fetched_same_day_issue_min": "20260101070",
                "fetched_same_day_issue_max": "20260101098",
                "fetched_same_day_issue_count": 29,
                "dynamic_context_n": 30,
                "runtime_history_issue_range": ["20260101098", "20260101098"],
            },
        }

    monkeypatch.setattr(
        api_module,
        "get_runtime",
        lambda: (
            FakeArtifacts(),
            {"history": {"min_dynamic_n": 20, "processed_path": "data/processed/history_processed.csv", "runtime_artifact_dir": "data/runtime_history"}, "auto_fetch": {"source": "x"}},
            None,
        ),
    )
    monkeypatch.setattr(api_module, "build_prediction_runtime_state", lambda artifacts, cfg: _FakeRuntimeState())
    monkeypatch.setattr(api_module, "run_prediction", fake_run_prediction)
    app.state.runtime_state = _FakeRuntimeState()

    client = TestClient(app)
    payload = {"recent_draws": [r.to_dict() for r in synthetic_records[-30:]]}

    first_result = {}

    def _first_call():
        first_result["response"] = client.post("/predict", json=payload)

    t = threading.Thread(target=_first_call)
    t.start()
    assert lock_entered.wait(timeout=1.0)

    second = client.post("/predict", json=payload)
    assert second.status_code == 429
    assert second.json()["detail"] == "prediction already running"

    release_lock.set()
    t.join(timeout=1.0)
    assert "response" in first_result
    assert first_result["response"].status_code == 200


def test_log_progress_request_id_and_flush(monkeypatch) -> None:
    captured = []

    def fake_print(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr("builtins.print", fake_print)
    log_progress(1, 6, "載入最近開獎上下文", "來源=manual", request_id="abcd1234")
    log_progress(1, 3, "舊格式測試", "detail")

    assert len(captured) == 2
    first_text = captured[0][0][0]
    assert first_text.startswith("[req=abcd1234] [進度] 1/6")
    assert "載入最近開獎上下文 | 來源=manual" in first_text
    assert captured[0][1]["flush"] is True

    second_text = captured[1][0][0]
    assert second_text.startswith("[進度] 1/3")
    assert "舊格式測試 | detail" in second_text
    assert captured[1][1]["flush"] is True


def test_load_recent_draws_passes_configured_timeout_to_consensus(monkeypatch, synthetic_records, tmp_path) -> None:
    observed: dict[str, float] = {}
    recent_two = [
        replace(synthetic_records[-2], issue="115016249", day_issue_index=1),
        replace(synthetic_records[-1], issue="115016250", day_issue_index=2),
    ]

    def fake_consensus(_sources, _report_path, mismatch_policy="fail_fast", timeout_s=10.0):
        observed["timeout_s"] = timeout_s
        return recent_two, {
            "consensus_status": "ok",
            "fetch_attempts": 1,
            "actual_source_used": "consensus_majority_merge",
            "source_same_day_max_issue": {"s1": "115016250", "s2": "115016250"},
        }

    monkeypatch.setattr("src.predict.run_source_consensus", fake_consensus)
    monkeypatch.setattr("src.predict.fetch_authoritative_latest_issue", lambda timeout_s=10.0: ("115016250", "probe"))
    config = {
        "auto_fetch": {
            "enabled": True,
            "sources": ["s1", "s2"],
            "fetch_timeout_seconds": 3.5,
            "consensus": {"on_mismatch": "majority_merge"},
        },
        "provenance": {"consensus_report_path": str(tmp_path / "consensus.json")},
    }
    _load_recent_draws(config, None)
    assert observed["timeout_s"] == 3.5


def test_load_recent_draws_uses_default_timeout_for_single_source(monkeypatch, synthetic_records) -> None:
    mocked = Mock()
    mocked.records = [
        replace(synthetic_records[-3], issue="115016248", day_issue_index=1),
        replace(synthetic_records[-2], issue="115016249", day_issue_index=2),
        replace(synthetic_records[-1], issue="115016250", day_issue_index=3),
    ]
    mocked.attempts = 1
    mocked.source_url = "s1"
    mocked.failover_reason = None
    observed: dict[str, float] = {}

    def fake_fetch_latest(*, sources, timeout_s=10.0):
        observed["timeout_s"] = timeout_s
        return mocked

    monkeypatch.setattr("src.predict.fetch_latest", fake_fetch_latest)
    monkeypatch.setattr("src.predict.fetch_authoritative_latest_issue", lambda timeout_s=10.0: ("115016250", "probe"))
    config = {
        "auto_fetch": {
            "enabled": True,
            "sources": ["s1"],
        }
    }
    _load_recent_draws(config, None)
    assert observed["timeout_s"] == 10.0
