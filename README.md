# BingoBingo Ranking Mainline

正式主線：`LightGBMRanker + LogisticRegression + 動態 N 視窗 retrieval + runtime rerank`。

## End-to-End Pipeline

1. 抓取今天目前已開盤（自動整理 day_issue_index=1..N）
   - `python -m src.fetch_winwin`
2. 匯入多年歷史 CSV（標準化）
   - `python -m src.prepare_data --inputs raw/賓果賓果_2024.csv raw/賓果賓果_2025.csv --output data/processed/history_processed.csv`
3. 建立 feature store（動態 N）
   - `python -m src.build_features --input data/processed/history_processed.csv --output data/feature_store/ranking_features.csv --min-history 100 --min-dynamic-n 20 --max-dynamic-n 999 --top-k 50`
4. 建立 ranking dataset（每期 80 candidates + group_id）
   - `python -m src.ranking_dataset --input data/feature_store/ranking_features.csv --output data/feature_store/ranking_dataset.csv`
5. 訓練（輸出 artifacts）
   - `python -m src.train --config configs/train.yaml --input data/feature_store/ranking_dataset.csv`
6. walk-forward 回測（含 fixed baseline / dynamic retrieval / dynamic fusion）
   - `python -m src.backtest --config configs/train.yaml --input data/feature_store/ranking_dataset.csv`
7. 預測（auto_fetch 會用今天第 1~N 期）
   - `python -m src.predict --config configs/predict.yaml --output reports/latest_prediction.json`
8. API
   - `uvicorn src.api:app --host 0.0.0.0 --port 8000`
9. Fast-path benchmark
   - `python -m src.benchmark_predict --config configs/predict.yaml --n 20 --warmup 3`

## Dynamic N Retrieval Contract

- 預測 context：
  - auto_fetch：`N = 今天目前已開盤數`。
  - manual recent_draws：`N = len(recent_draws)`。
- retrieval：歷史中找所有連續長度 `N` 的視窗，比對多子分數後取 top-k。
- 輸出分數明確區分 `ranking_score` 與 `auxiliary_score`，不假裝機率。

## Required Model Artifacts

`models/` 內至少需有：
- `lightgbm_ranker.txt`
- `logistic_regression.pkl`
- `feature_columns.json`
- `metadata.json`

API 啟動與預測均會檢查以上檔案，缺任一檔將 fail-fast。

## Predict Fast Path (Startup preload + cache)

- FastAPI startup 會預載：
  - model artifacts
  - runtime history artifact / store
  - merged history snapshot
  - retrieval precomputed index（indicator/profile/prefix）
  - recent cache（含 recent_hash / latest issue）
- `/predict` warm path 只做：
  - cache resolve
  - prepared retrieval lookup
  - feature contract + model predict + runtime rerank
- 觀測資訊：
  - `/health`：model/retrieval/cache readiness
  - `/debug/runtime`：runtime index/cache lightweight diagnostics

## Testing

- `flake8 agent.py`
- `flake8 src tests`
- `pytest -q`

## 驗收方式（最少一組）

1. 確認使用「今天第 1~N 期」：
   - 執行 `python -m src.fetch_winwin`，檢查 `data/raw/winwin_latest_processed.csv` 的 `day_issue_index` 是否從 1 連續到最後一筆。
2. 確認 retrieval 走「歷史 N 期視窗比對」：
   - 執行 `python -m src.predict --config configs/predict.yaml --output reports/latest_prediction.json`，檢查輸出 `dynamic_context_n` 與 `retrieval_top_matches`。
3. 確認排序有吃到 `retrieval_next_draw_vote`：
   - 查看 `ranking_score_table` 的 `retrieval_score`（來源為 `retrieval_next_draw_posterior`）是否影響 `final_score` 排序。


## Provenance / Audit Outputs

- `reports/raw_manifest.json`
- `reports/local_data_audit.json`
- `reports/source_consensus_report.json`（多來源時）
- `reports/history_snapshot.json`
- `reports/predictability_test.json`
- `reports/permutation_distribution.csv`
- `reports/block_bootstrap_summary.json`
- `reports/alignment_audit.json`


## Large-file Safe Output Contract

Mainline now writes large tabular outputs through `src.io_utils.safe_write_table` and reads through `safe_read_table`:

- Soft threshold: `95MB` per file (auto shard).
- Hard limit: `100MB` per file (`>=100MB` fail-fast).
- Supported read formats: `.csv`, `.csv.gz`, `.parquet`, dataset directory (`*.dataset` with `manifest.json`), and direct `manifest.json`.
- When sharded, output directory includes `manifest.json` with: `format`, `compression`, `shard_count`, `shards`, `columns`, `row_count`, `created_at`, `producer_script`.
- CLI usage remains unchanged; if output path is a legacy `.csv`, writer may materialize a sibling `*.dataset` directory and downstream mainline still reads it.

Affected pipeline stages:

1. `python -m src.fetch_winwin` (supports `--today-only`, `--gzip`, `--max-file-mb`)
2. `python -m src.prepare_data ...`
3. `python -m src.build_features ...`
4. `python -m src.ranking_dataset ...`
5. `python -m src.train ... --max-file-mb 95`
