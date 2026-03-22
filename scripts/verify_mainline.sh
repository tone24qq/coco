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
pytest -q tests/test_agents/test_phase1_predict_schema.py
pytest -q tests/test_agents/test_phase2_api_and_pipeline.py
pytest -q tests/test_agents/test_phase3_provenance_and_consensus.py
pytest -q tests/test_agents/test_phase3_snapshot_predict_backtest.py
pytest -q tests/test_agents/test_phase1_backtest_runtime_parity.py
pytest -q tests/test_agents/test_phase4_ranking_mainline_contracts.py
pytest -q tests/test_agents/test_phase5_failover_size_and_logs.py
pytest -q tests/test_agents/test_phase5_dynamic_weighting.py
pytest -q tests/test_agents/test_phase6_runtime_history.py
pytest -q tests/test_agents/test_phase7_large_file_io.py
pytest -q tests/test_agents/test_phase8_retrieval_vectorized_equivalence.py
pytest -q tests/test_agents/test_phase8_build_features_vectorized_equivalence.py
pytest -q tests/test_agents/test_phase8_compute_metrics_equivalence.py
pytest -q tests/test_agents/test_phase8_oof_cv_equivalence.py
pytest -q tests/test_agents/test_phase8_perf_sanity.py
pytest -q tests/test_agents/test_phase9_autofetch_contract.py

echo "[5/7] prepare minimal ranking dataset"
python - <<'PY'
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.build_features import build_feature_rows, write_feature_store
from src.io_utils import safe_read_table
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
feature_out = write_feature_store(feature_path, rows)
rank_rows = safe_read_table(feature_out).to_dict(orient="records")
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
rm -rf data/runtime_history
python -m src.predict --config configs/predict.yaml --output reports/latest_prediction.json --recent-json /tmp/verify_recent.json > reports/logs/predict.log 2>&1
bash scripts/build_deploy_bundle.sh /tmp/deploy_bundle_verify > reports/logs/build_deploy_bundle.log 2>&1
python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history
rm data/processed/history_processed.csv
python -m src.predict --config configs/predict.yaml --output reports/latest_prediction_artifact_only.json --recent-json /tmp/verify_recent.json > reports/logs/predict_artifact_only.log 2>&1

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
artifact_only_pred = Path("reports/latest_prediction_artifact_only.json")
logs = [
    Path("reports/logs/build_features.log"),
    Path("reports/logs/train.log"),
    Path("reports/logs/backtest.log"),
    Path("reports/logs/predict.log"),
    Path("reports/logs/predict_artifact_only.log"),
    Path("reports/logs/build_deploy_bundle.log"),
]
progress_logs = [
    Path("reports/logs/build_features.log"),
    Path("reports/logs/train.log"),
    Path("reports/logs/backtest.log"),
    Path("reports/logs/predict.log"),
    Path("reports/logs/predict_artifact_only.log"),
]

for p in [train_reg, train_fold, backtest_fold, backtest_summary, alignment, bootstrap, predictability, perm, latest_pred, artifact_only_pred, *logs]:
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
pred_artifact_only = json.loads(artifact_only_pred.read_text(encoding="utf-8"))
assert len(pred["ranking_score_table"]) == 80
assert len(pred["top20_numbers"]) == 20
assert pred["metadata"]["score_type"] == "ranking_score"
assert pred["metadata"]["target_next_issue_contract"] == "passed"
if pred["metadata"].get("fetched_same_day_issue_max") is not None:
    assert pred["metadata"]["latest_fetched_issue"] == pred["metadata"]["fetched_same_day_issue_max"]
assert pred["issue"] == str(int(pred["metadata"]["latest_fetched_issue"]) + 1)
for key in ["fetch", "merge", "retrieval_feature_build", "model_score", "total"]:
    assert key in pred["metadata"]["elapsed_ms"]
    assert pred["metadata"]["elapsed_ms"][key] >= 0.0
assert "freshness_probe" in pred["metadata"]["elapsed_ms"]
assert pred["metadata"]["elapsed_ms"]["freshness_probe"] >= 0.0
assert "dynamic_weighting" in pred["metadata"]
assert "effective_runtime_weights" in pred["metadata"]
assert abs(sum(pred["metadata"]["effective_runtime_weights"].values()) - 1.0) <= 1e-6
for key in ["top20_numbers", "top10_numbers", "top3_numbers", "ranking_score_table", "retrieval_top_matches"]:
    assert pred_artifact_only[key] == pred[key], f"artifact-only prediction drift on {key}"

for p in [train_reg, train_fold, backtest_fold, backtest_summary, alignment, bootstrap, predictability, perm, latest_pred, artifact_only_pred]:
    assert p.stat().st_size <= 100 * 1024 * 1024, f"file exceeds 100MB: {p}"

for p in progress_logs:
    txt = p.read_text(encoding="utf-8")
    assert "進度" in txt, f"missing Chinese progress log: {p}"
    assert "%" in txt, f"missing percent progress log: {p}"
for p in logs:
    assert p.stat().st_size <= 100 * 1024 * 1024, f"log exceeds 100MB: {p}"

for f in ["meta.json", "numbers.npy", "issue.npy", "draw_date_ordinal.npy", "day_issue_index.npy"]:
    assert Path(f"/tmp/deploy_bundle_verify/data/runtime_history/{f}").exists(), f"deploy bundle missing runtime artifact: {f}"
PY


python - <<'PY'
from pathlib import Path
import json

tracked = [
    Path("data/processed/history_processed.csv"),
    Path("data/processed/history_processed.parquet"),
    Path("data/processed/history_processed.dataset"),
    Path("data/feature_store/ranking_features.csv"),
    Path("data/feature_store/ranking_features.parquet"),
    Path("data/feature_store/ranking_features.dataset"),
    Path("data/feature_store/ranking_dataset.csv"),
    Path("data/feature_store/ranking_dataset.parquet"),
    Path("data/feature_store/ranking_dataset.dataset"),
]

for p in tracked:
    if p.exists() and p.is_file():
        print(f"[size] {p} -> {p.stat().st_size}")
        assert p.stat().st_size < 100 * 1024 * 1024
    if p.exists() and p.is_dir():
        manifest = p / "manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            print(f"[manifest] {manifest}: format={payload.get('format')} shards={payload.get('shard_count')}")
            for name in payload.get("shards", []):
                s = p / name
                assert s.exists()
                assert s.stat().st_size < 100 * 1024 * 1024
PY

echo "verify_mainline.sh PASSED"
