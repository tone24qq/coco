import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from src.fetch_winwin import AUZO_URL, WINWIN_URL, fetch_latest
from src.ranking_dataset import load_feature_rows
from src.utils import DataContractError, enforce_file_size


def test_fetch_latest_failover_to_auzo(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, text: str, ok: bool = True):
            self.text = text
            self._ok = ok

        def raise_for_status(self):
            if not self._ok:
                raise RuntimeError("bad status")

    calls = []

    def fake_get(url, timeout=10.0, params=None):
        calls.append((url, params))
        if url == WINWIN_URL:
            raise RuntimeError("primary down")
        html = "<table><tr><td>20260320011</td><td>2026/03/20 10:10:00</td>" + "".join([f"<td>{i}</td>" for i in range(1, 21)]) + "</tr></table>"
        return DummyResponse(html)

    monkeypatch.setattr("src.fetch_winwin.httpx.get", fake_get)
    out = fetch_latest([WINWIN_URL, AUZO_URL])
    assert out.source_url == AUZO_URL
    assert out.failover_reason is not None
    assert calls[0][0] == WINWIN_URL and calls[1][0] == AUZO_URL


def test_fetch_latest_winwin_dynamic_fallback(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, text: str = "", payload=None):
            self.text = text
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, timeout=10.0, params=None):
        if url == WINWIN_URL:
            html = '<html><div id="bingoTable"></div><script>loadBingoData("2026-03-20")</script></html>'
            return DummyResponse(text=html)
        assert url.endswith('/Bingo/GetBingoData')
        assert params is not None and 'date' in params
        payload = {
            "Data": [
                {
                    "No": "20260320001",
                    "OpenDate": "2026-03-20 09:05:00",
                    "BigShowOrder": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
                    "HighLowTop": "10:10",
                    "OddEvenTop": "10:10",
                }
            ]
        }
        return DummyResponse(payload=payload)

    monkeypatch.setattr("src.fetch_winwin.httpx.get", fake_get)
    out = fetch_latest([WINWIN_URL])
    assert out.source_url == WINWIN_URL
    assert out.records[0].issue == "20260320001"
    assert out.records[0].day_issue_index == 1


def test_fetch_latest_all_sources_fail_fast(monkeypatch) -> None:
    def bad_get(url, timeout=10.0, params=None):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.fetch_winwin.httpx.get", bad_get)
    with pytest.raises(DataContractError):
        fetch_latest([WINWIN_URL, AUZO_URL])


def test_sharded_feature_and_ranking_loader(tmp_path: Path) -> None:
    feature_path = tmp_path / "ranking_features.csv"
    df = pd.DataFrame(
        [
            {
                "issue": "I1",
                "draw_date": "2026-01-01",
                "candidate_number": i,
                "label": int(i <= 20),
                "group_id": 0,
            }
            for i in range(1, 81)
        ]
    )
    part1 = tmp_path / "ranking_features.part0001.csv"
    part2 = tmp_path / "ranking_features.part0002.csv"
    df.iloc[:40].to_csv(part1, index=False)
    df.iloc[40:].to_csv(part2, index=False)

    rows = load_feature_rows(feature_path)
    assert len(rows) == 80


def test_file_size_fail_fast_gate(tmp_path: Path) -> None:
    p = tmp_path / "too_large.bin"
    p.write_bytes(b"0" * 1024)
    with pytest.raises(DataContractError):
        enforce_file_size(p, max_bytes=16)


def test_train_and_backtest_emit_chinese_progress(monkeypatch, tmp_path: Path, ranking_dataset_path: Path, capsys) -> None:
    import src.train as train_module
    import src.backtest as backtest_module

    fake_scored = pd.DataFrame(
        {
            "issue": ["I1", "I1", "I2", "I2"],
            "candidate_number": [1, 2, 1, 2],
            "label": [1, 0, 0, 1],
            "final_score": [0.9, 0.1, 0.2, 0.8],
            "ranker_score": [0.9, 0.1, 0.2, 0.8],
            "logistic_score": [0.8, 0.2, 0.3, 0.7],
            "retrieval_score": [0.7, 0.3, 0.3, 0.7],
            "history_prior_score": [0.6, 0.4, 0.4, 0.6],
            "analysis_rerank_score": [0.5, 0.5, 0.5, 0.5],
            "local_peak_score": [0.5, 0.5, 0.5, 0.5],
            "cand_hits_last_100": [10, 1, 1, 10],
            "cand_hits_last_20": [3, 1, 1, 3],
            "retrieval_top3_hit_flag": [1, 0, 0, 1],
            "retrieval_exact_window_match_count": [0, 0, 0, 0],
            "retrieval_exact_draw_match_count_mean": [1, 0, 0, 1],
        }
    )

    def fake_run_cv(*args, **kwargs):
        return [SimpleNamespace(fold_id=1, train_scored=fake_scored.copy(), val_scored=fake_scored.copy(), train_issues=["I0"], val_issues=["I1", "I2"])]

    monkeypatch.setattr(train_module, "run_cv", fake_run_cv)
    monkeypatch.setattr(backtest_module, "run_cv", fake_run_cv)
    monkeypatch.setattr(train_module, "build_canonical_audit", lambda **kwargs: ({}, []))

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump({"validation": {"n_splits": 1, "min_train_issues": 1}, "runtime_scoring": {"weights": {}}, "history": {"processed_path": str(tmp_path / "hp.csv")}}), encoding="utf-8")
    exp_path = tmp_path / "experiments.yaml"
    exp_path.write_text(yaml.safe_dump({"experiments": [{"name": "baseline_frequency"}, {"name": "ranker_main_qsm"}]}), encoding="utf-8")
    (tmp_path / "hp.csv").write_text("issue,draw_date,numbers,day_issue_index\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cfg_path), "--experiments", str(exp_path), "--input", str(ranking_dataset_path)])
    train_module.main()
    monkeypatch.setattr(sys, "argv", ["backtest", "--config", str(cfg_path), "--experiments", str(exp_path), "--input", str(ranking_dataset_path)])
    backtest_module.main()
    output = capsys.readouterr().out
    assert "進度" in output
    assert "%" in output
