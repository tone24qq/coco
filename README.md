# 1-80 Candidate Ranking System (Transformer Mainline)

正式入口：
- 訓練：`python -m src.train_transformer --input <history.csv> --output models/transformer_v1`
- 回測：`python -m src.backtest_transformer --input <history.csv> --output reports/transformer_backtest`
- CLI 預測：`python -m src.predict --runtime-dir data/runtime_history`
- API 預測：`GET /predict`（`app.py`）

## Deploy contract（保留）
- `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
- `bash scripts/build_deploy_bundle.sh deploy_bundle`

## Runtime artifact
`runtime_history` 僅建置/同步 artifact，不重訓：
- history: `history_runtime.parquet` + `history_runtime.csv`
- score chain: `scores.parquet` + `scores.csv`
- model: `model.ckpt`, `transformer_metadata.json`
- contract metadata: `metadata.json`

## History merge behavior
- `merge_history` 允許 issue 缺期（非連號）資料合併，只要求 issue 嚴格遞增且不可重複。
- 若 local/latest 在同 issue 有任一欄位不一致，維持 fail-fast（issue conflict）。
- 目前 gap / rolling / retrieval 視窗以 observed rows 計算，不代表真實連續 issue 距離。

## Predict response
- `latest_known_issue`, `target_issue`
- `model_version`, `feature_version`
- `data_source`, `fetch_attempts`
- `source_latest_issues`, `selected_source_reason`, `source_records_count`
- `source_tail_count`, `selected_source_full_records_count`, `selected_source_tail_count`
- `consensus_status`, `max_observed_issue`
- `raw_scores`, `raw_top20`, `raw_top3`
- `final_top20`, `final_top3`, `rerank_applied`, `rerank_reason`
- `score_type: ranking_score`
- `scores`, `top20`, `top3`
- `diversity_relaxed`
- `drift_metadata`
- `stale_issues`, `is_stale`

詳細 contract 見 `ARCHITECTURE.md`。
