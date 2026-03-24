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

## Predict response
- `latest_known_issue`, `target_issue`
- `model_version`, `feature_version`
- `data_source`, `fetch_attempts`
- `score_type: ranking_score`
- `scores`, `top20`, `top3`
- `diversity_relaxed`
- `drift_metadata`
- `stale_issues`, `is_stale`

詳細 contract 見 `ARCHITECTURE.md`。
