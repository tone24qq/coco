from datetime import date
from pathlib import Path

import pytest

from src.fetch_winwin import FetchResult
from src.fetchers.source_consensus import run_source_consensus
from src.io.canonical_dataset import build_canonical_audit
from src.utils import DataContractError, DrawRecord


def _write_raw(path: Path, rows: list[DrawRecord]) -> None:
    header = "期別,開獎日期," + ",".join([f"獎號{i}" for i in range(1, 21)])
    lines = [header]
    for r in rows:
        lines.append(f"{r.issue},{r.draw_date.strftime('%Y/%m/%d')}," + ",".join(str(x) for x in r.numbers))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_raw_manifest_and_audit_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        DrawRecord(issue="20260101001", draw_date=date(2026, 1, 1), numbers=tuple(range(1, 21)), day_issue_index=1),
        DrawRecord(issue="20260101002", draw_date=date(2026, 1, 1), numbers=tuple(range(2, 22)), day_issue_index=2),
    ]
    _write_raw(raw_dir / "hist.csv", rows)
    audit, merged = build_canonical_audit(
        raw_dirs=[raw_dir],
        audit_output_path=tmp_path / "reports" / "local_data_audit.json",
        manifest_output_path=tmp_path / "reports" / "raw_manifest.json",
    )
    assert audit["file_count"] == 1
    assert audit["total_rows"] == 2
    assert len(merged) == 2
    assert (tmp_path / "reports" / "raw_manifest.json").exists()
    assert (tmp_path / "reports" / "local_data_audit.json").exists()


def test_raw_header_fail_fast(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "bad.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(DataContractError):
        build_canonical_audit(raw_dirs=[raw_dir], audit_output_path=tmp_path / "a.json", manifest_output_path=tmp_path / "m.json")


def test_source_consensus_ok_and_mismatch(monkeypatch, synthetic_records) -> None:
    recs_a = synthetic_records[-5:]
    recs_b = synthetic_records[-5:]
    recs_c = synthetic_records[-5:].copy()
    recs_c[-1] = DrawRecord(
        issue=recs_c[-1].issue,
        draw_date=recs_c[-1].draw_date,
        numbers=tuple(sorted(set(list(recs_c[-1].numbers)[:-1] + [80]))),
        day_issue_index=recs_c[-1].day_issue_index,
    )

    def fake_fetch_latest(sources=None, timeout_s=10.0):
        src = sources[0]
        if src == "s1":
            return FetchResult(recs_a, src, "t", 1, failover_reason="primary_down")
        if src == "s2":
            return FetchResult(recs_b, src, "t", 1)
        return FetchResult(recs_c, src, "t", 1)

    monkeypatch.setattr("src.fetchers.source_consensus.fetch_latest", fake_fetch_latest)
    _, report_ok = run_source_consensus(["s1", "s2"], report_path=Path("reports/test_consensus_ok.json"))
    assert report_ok["consensus_status"] in {"ok", "partial"}
    assert report_ok["successful_sources"] == ["s1", "s2"]
    assert report_ok["failover_reason"] == "primary_down"

    with pytest.raises(DataContractError):
        run_source_consensus(["s1", "s3"], report_path=Path("reports/test_consensus_bad.json"), mismatch_policy="fail_fast")
    _, report_bad = run_source_consensus(
        ["s1", "s3"], report_path=Path("reports/test_consensus_bad_majority.json"), mismatch_policy="majority_merge"
    )
    assert report_bad["consensus_status"] == "mismatch"
    assert report_bad["merge_strategy"] == "majority_merge"


def test_source_consensus_partial(monkeypatch, synthetic_records) -> None:
    recs_a = synthetic_records[-5:]
    recs_b = synthetic_records[-4:]

    def fake_fetch_latest(sources=None, timeout_s=10.0):
        src = sources[0]
        if src == "s1":
            return FetchResult(recs_a, src, "t", 1)
        return FetchResult(recs_b, src, "t", 1)

    monkeypatch.setattr("src.fetchers.source_consensus.fetch_latest", fake_fetch_latest)
    _, report = run_source_consensus(["s1", "s2"], report_path=Path("reports/test_consensus_partial.json"), mismatch_policy="majority_merge")
    assert report["consensus_status"] == "partial"


def test_all_sources_fail_fast(monkeypatch) -> None:
    def bad_fetch_latest(sources=None, timeout_s=10.0):
        raise DataContractError("boom")

    monkeypatch.setattr("src.fetchers.source_consensus.fetch_latest", bad_fetch_latest)
    with pytest.raises(DataContractError):
        run_source_consensus(["a", "b"], report_path=Path("reports/test_consensus_fail.json"))
