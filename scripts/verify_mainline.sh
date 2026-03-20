#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] lint"
flake8 agent.py
flake8 src tests

echo "[2/7] compile"
python -m py_compile $(git ls-files '*.py')

echo "[3/7] unit tests"
pytest -q

echo "[4/7] targeted tests"
pytest -q tests/test_agents/test_phase1_fetch.py
pytest -q tests/test_agents/test_phase3_provenance_and_consensus.py
pytest -q tests/test_agents/test_phase3_snapshot_predict_backtest.py
pytest -q tests/test_agents/test_phase1_backtest_runtime_parity.py
pytest -q tests/test_agents/test_phase4_ranking_mainline_contracts.py
pytest -q tests/test_agents/test_phase5_failover_size_and_logs.py
pytest -q tests/test_agents/test_phase5_dynamic_weighting.py
pytest -q tests/test_agents/test_phase6_runtime_history.py

echo "[5/7] prepare minimal ranking dataset"
python - <<'PY'
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.build_features import build_feature_rows, write_feature_store
from src.ranking_dataset import attach_group_ids, write_rows
from src.utils import DrawRecord, write_processed

records = []
for i in range(600):
    start = (i % 60) + 1
    nums = tuple(sorted(((start + k - 1) % 80) + 1 for k in range(20)))
    records.append(
        DrawRecord(
            issue=f"{20260101000 + i}",
            draw_date=date(2026, 1, 1) + timedelta(days=i // 30),
            numbers=nums,
            day_issue_index=(i % 30) + 1,
        )
    )
rows = build_feature_rows(records, min_history=120, retrieval_window=40, top_k=8)
feature_path = Path("data/feature_store/ranking_features.csv")
feature_path.parent.mkdir(parents=True, exist_ok=True)
write_feature_store(feature_path, rows)
rank_rows = pd.read_csv(feature_path).to_dict(orient="records")
write_rows(Path("data/feature_store/ranking_dataset.csv"), attach_group_ids(rank_rows))
write_processed(Path("data/processed/history_processed.csv"), records)
PY

mkdir -p reports/logs

echo "[6/9] build_features mainline execution"
python -m src.build_features --input data/processed/history_processed.csv --output data/feature_store/ranking_features.csv --min-history 120 --min-dynamic-n 20 --max-dynamic-n 40 --top-k 8 > reports/logs/build_features.log 2>&1

echo "[7/9] train mainline execution"
python -m src.train --config configs/train.yaml --experiments configs/experiments.yaml --input data/feature_store/ranking_dataset.csv > reports/logs/train.log 2>&1

echo "[8/9] backtest mainline execution"
python -m src.backtest --config configs/train.yaml --experiments configs/experiments.yaml --input data/feature_store/ranking_dataset.csv > reports/logs/backtest.log 2>&1

echo "[8.5/9] failover behavior check"
python - <<'PY'
import json
from datetime import date
from pathlib import Path
from src.fetch_winwin import FetchResult, WINWIN_URL, AUZO_URL
import src.fetchers.source_consensus as sc
from src.utils import DrawRecord

rows = [
    DrawRecord(issue="20260101001", draw_date=date(2026, 1, 1), numbers=tuple(range(1, 21)), day_issue_index=1),
    DrawRecord(issue="20260101002", draw_date=date(2026, 1, 1), numbers=tuple(range(2, 22)), day_issue_index=2),
]

def fake_fetch_latest(sources=None, timeout_s=10.0):
    src = (sources or [WINWIN_URL])[0]
    if src == WINWIN_URL:
        raise RuntimeError("primary down")
    return FetchResult(rows, AUZO_URL, "2026-01-01T00:00:00", 2, failover_reason="primary_down")

old = sc.fetch_latest
sc.fetch_latest = fake_fetch_latest
try:
    _, report = sc.run_source_consensus([WINWIN_URL, AUZO_URL], report_path=Path("reports/source_consensus_report.json"), mismatch_policy="majority_merge")
    assert report["successful_sources"] == [AUZO_URL]
    assert report["failover_reason"] is not None
finally:
    sc.fetch_latest = old
PY

echo "[9/9] predict mainline execution"
python - <<'PY'
import json
from datetime import date, timedelta
from pathlib import Path
from src.utils import DrawRecord

records = []
for i in range(600):
    start = (i % 60) + 1
    nums = tuple(sorted(((start + k - 1) % 80) + 1 for k in range(20)))
    records.append(
        DrawRecord(
            issue=f"{20260101000 + i}",
            draw_date=date(2026, 1, 1) + timedelta(days=i // 30),
            numbers=nums,
            day_issue_index=(i % 30) + 1,
        )
    )
recent = [r.to_dict() for r in records[-30:]]
Path("/tmp/verify_recent.json").write_text(json.dumps(recent, ensure_ascii=False), encoding="utf-8")
PY
python -m src.predict --config configs/predict.yaml --output reports/latest_prediction.json --recent-json /tmp/verify_recent.json > reports/logs/predict.log 2>&1

echo "[10/10] output assertions"
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

train_reg = Path('reports/train_experiment_registry.csv')
train_fold = Path('reports/train_experiment_per_fold_metrics.csv')
backtest_fold = Path('reports/backtest_experiment_per_fold_metrics.csv')
backtest_summary = Path('reports/backtest_experiment_summary.json')
alignment = Path('reports/alignment_audit.json')
bootstrap = Path('reports/block_bootstrap_summary.json')
predictability = Path('reports/predictability_test.json')
perm = Path('reports/permutation_distribution.csv')
latest_pred = Path('reports/latest_prediction.json')
logs = [Path("reports/logs/build_features.log"), Path("reports/logs/train.log"), Path("reports/logs/backtest.log"), Path("reports/logs/predict.log")]

for p in [train_reg, train_fold, backtest_fold, backtest_summary, alignment, bootstrap, predictability, perm, latest_pred, *logs]:
    assert p.exists(), f'missing output: {p}'

reg_df = pd.read_csv(train_reg)
assert not reg_df.empty and (reg_df['status'] == 'completed').all()

train_df = pd.read_csv(train_fold)
backtest_df = pd.read_csv(backtest_fold)
assert train_df['experiment'].nunique() >= 2
assert backtest_df['experiment'].nunique() >= 2

summary = json.loads(backtest_summary.read_text(encoding='utf-8'))
assert summary['train_vs_backtest_gap_top3'] != 'unavailable'

align = json.loads(alignment.read_text(encoding='utf-8'))
assert 'time_series_split_forward_only' in align
assert 'runtime_backtest_scoring_formula_match' in align

boot = json.loads(bootstrap.read_text(encoding='utf-8'))
assert boot['metric'] == 'mainline_minus_baseline_top3'

test = json.loads(predictability.read_text(encoding='utf-8'))
assert test['null_hypothesis'] == 'mainline_minus_baseline_mean<=0'

perm_df = pd.read_csv(perm)
assert not perm_df.empty

pred = json.loads(latest_pred.read_text(encoding='utf-8'))
assert len(pred["ranking_score_table"]) == 80
assert len(pred["top20_numbers"]) == 20
assert pred["metadata"]["score_type"] == "ranking_score"
assert pred["metadata"]["target_next_issue_contract"] == "passed"
assert "dynamic_weighting" in pred["metadata"]
assert "effective_runtime_weights" in pred["metadata"]
assert abs(sum(pred["metadata"]["effective_runtime_weights"].values()) - 1.0) <= 1e-6

for p in [train_reg, train_fold, backtest_fold, backtest_summary, alignment, bootstrap, predictability, perm, latest_pred]:
    assert p.stat().st_size <= 100 * 1024 * 1024, f"file exceeds 100MB: {p}"

for p in logs:
    txt = p.read_text(encoding="utf-8")
    assert "進度" in txt, f"missing Chinese progress log: {p}"
    assert "%" in txt, f"missing percent progress log: {p}"
    assert p.stat().st_size <= 100 * 1024 * 1024, f"log exceeds 100MB: {p}"
PY

echo "verify_mainline.sh PASSED"
