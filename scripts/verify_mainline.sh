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
pytest -q tests/test_agents/test_phase3_provenance_and_consensus.py
pytest -q tests/test_agents/test_phase3_snapshot_predict_backtest.py
pytest -q tests/test_agents/test_phase1_backtest_runtime_parity.py

echo "[5/7] prepare minimal ranking dataset"
python - <<'PY'
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.build_features import build_feature_rows, write_feature_store
from src.ranking_dataset import attach_group_ids, write_rows
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
rows = build_feature_rows(records, min_history=120, retrieval_window=40, top_k=8)
feature_path = Path("data/feature_store/ranking_features.csv")
feature_path.parent.mkdir(parents=True, exist_ok=True)
write_feature_store(feature_path, rows)
rank_rows = pd.read_csv(feature_path).to_dict(orient="records")
write_rows(Path("data/feature_store/ranking_dataset.csv"), attach_group_ids(rank_rows))
PY

echo "[6/7] train mainline execution"
python -m src.train --config configs/train.yaml --experiments configs/experiments.yaml --input data/feature_store/ranking_dataset.csv

echo "[7/7] backtest mainline execution"
python -m src.backtest --config configs/train.yaml --experiments configs/experiments.yaml --input data/feature_store/ranking_dataset.csv

echo "[8/8] output assertions"
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

for p in [train_reg, train_fold, backtest_fold, backtest_summary, alignment, bootstrap, predictability, perm]:
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
PY

echo "verify_mainline.sh PASSED"
