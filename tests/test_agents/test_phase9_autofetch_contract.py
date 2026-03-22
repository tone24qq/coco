from datetime import date
from pathlib import Path

import pytest

from src.predict import _load_recent_draws
from src.utils import DataContractError, DrawRecord


def _same_day_rows(day_prefix: str, end_idx: int) -> list[DrawRecord]:
    rows: list[DrawRecord] = []
    for idx in range(1, end_idx + 1):
        issue = f"{day_prefix}{idx:03d}"
        start = ((idx - 1) % 60) + 1
        numbers = tuple(sorted(((start + k - 1) % 80) + 1 for k in range(20)))
        rows.append(DrawRecord(issue=issue, draw_date=date(2026, 3, 22), numbers=numbers, day_issue_index=idx))
    return rows


def _cfg(tmp_path: Path) -> dict:
    return {
        "auto_fetch": {
            "enabled": True,
            "sources": ["s1", "s2"],
            "consensus": {"on_mismatch": "majority_merge"},
        },
        "provenance": {"consensus_report_path": str(tmp_path / "consensus.json")},
    }


def test_auto_fetch_target_uses_latest_same_day_250(monkeypatch, tmp_path) -> None:
    rows = _same_day_rows("115016", 250)

    def fake_consensus(_sources, _report_path, mismatch_policy="majority_merge", timeout_s=10.0):
        return rows, {
            "consensus_status": "ok",
            "fetch_attempts": 2,
            "actual_source_used": "consensus_majority_merge",
            "source_same_day_max_issue": {"s1": "115016250", "s2": "115016250"},
        }

    monkeypatch.setattr("src.predict.run_source_consensus", fake_consensus)
    monkeypatch.setattr("src.predict.fetch_authoritative_latest_issue", lambda timeout_s=10.0: ("115016250", "probe"))
    recent, _source, meta = _load_recent_draws(_cfg(tmp_path), None)

    assert recent[-1].issue == "115016250"
    assert meta["fetched_same_day_issue_max"] == "115016250"
    assert meta["verified_latest_fetched_issue"] == "115016250"
    assert meta["freshness_check_passed"] is True


def test_auto_fetch_accepts_mixed_sources_if_merged_to_250(monkeypatch, tmp_path) -> None:
    rows = _same_day_rows("115016", 250)

    def fake_consensus(_sources, _report_path, mismatch_policy="majority_merge", timeout_s=10.0):
        return rows, {
            "consensus_status": "partial",
            "fetch_attempts": 2,
            "actual_source_used": "consensus_majority_merge",
            "source_same_day_max_issue": {"s1": "115016240", "s2": "115016250"},
        }

    monkeypatch.setattr("src.predict.run_source_consensus", fake_consensus)
    monkeypatch.setattr("src.predict.fetch_authoritative_latest_issue", lambda timeout_s=10.0: ("115016250", "probe"))
    recent, _source, meta = _load_recent_draws(_cfg(tmp_path), None)

    assert recent[-1].issue == "115016250"
    assert meta["source_same_day_max_issue"]["s1"] == "115016240"
    assert meta["source_same_day_max_issue"]["s2"] == "115016250"


def test_auto_fetch_fail_fast_when_all_sources_partial_same_day(monkeypatch, tmp_path) -> None:
    # missing a same-day middle slice should fail completeness contract immediately
    rows = _same_day_rows("115016", 240)

    def fake_consensus(_sources, _report_path, mismatch_policy="majority_merge", timeout_s=10.0):
        gapped = rows[:220] + rows[225:]
        return gapped, {
            "consensus_status": "partial",
            "fetch_attempts": 2,
            "actual_source_used": "consensus_majority_merge",
            "source_same_day_max_issue": {"s1": "115016240", "s2": "115016240"},
        }

    monkeypatch.setattr("src.predict.run_source_consensus", fake_consensus)
    monkeypatch.setattr("src.predict.fetch_authoritative_latest_issue", lambda timeout_s=10.0: ("115016240", "probe"))
    with pytest.raises(DataContractError, match="incomplete issue set"):
        _load_recent_draws(_cfg(tmp_path), None)


def test_auto_fetch_fail_fast_when_all_sources_stale_against_authoritative(monkeypatch, tmp_path) -> None:
    rows = _same_day_rows("115016", 240)

    def fake_consensus(_sources, _report_path, mismatch_policy="majority_merge", timeout_s=10.0):
        return rows, {
            "consensus_status": "ok",
            "fetch_attempts": 2,
            "actual_source_used": "consensus_majority_merge",
            "source_same_day_max_issue": {"s1": "115016240", "s2": "115016240"},
        }

    monkeypatch.setattr("src.predict.run_source_consensus", fake_consensus)
    monkeypatch.setattr("src.predict.fetch_authoritative_latest_issue", lambda timeout_s=10.0: ("115016250", "probe"))
    with pytest.raises(DataContractError, match="merged max issue=115016240, authoritative latest issue=115016250"):
        _load_recent_draws(_cfg(tmp_path), None)
