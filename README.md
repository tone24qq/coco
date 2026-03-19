# BingoBingo Ranking Mainline

正式主線：`LightGBMRanker + LogisticRegression + 歷史相似視窗 retrieval + runtime rerank`。

## End-to-End Pipeline

1. 抓取最新資料
   - `python -m src.fetch_winwin`
2. 匯入多年歷史 CSV（標準化）
   - `python -m src.prepare_data --inputs raw/賓果賓果_2024.csv raw/賓果賓果_2025.csv --output data/processed/history_processed.csv`
3. 建立 feature store
   - `python -m src.build_features --input data/processed/history_processed.csv --output data/feature_store/ranking_features.csv`
4. 建立 ranking dataset（每期 80 candidates + group_id）
   - `python -m src.ranking_dataset --input data/feature_store/ranking_features.csv --output data/feature_store/ranking_dataset.csv`
5. 訓練（輸出 artifacts）
   - `python -m src.train --config configs/train.yaml --input data/feature_store/ranking_dataset.csv`
6. walk-forward 回測
   - `python -m src.backtest --config configs/train.yaml --input data/feature_store/ranking_dataset.csv`
7. 預測
   - `python -m src.predict --config configs/predict.yaml --output reports/latest_prediction.json`
8. API
   - `uvicorn src.api:app --host 0.0.0.0 --port 8000`

## Contract

- Ranking contract：每個 issue 必須 80 candidates。
- 嚴禁洩漏：train/backtest 使用 time-series split，predict 僅使用過去資料。
- Fail-fast：缺資料、缺 artifact、feature contract 不符時立即報錯。
- 輸出分數明確區分 `ranking_score` 與 `auxiliary_score`，不假裝機率。

## Required Model Artifacts

`models/` 內至少需有：
- `lightgbm_ranker.txt`
- `logistic_regression.pkl`
- `feature_columns.json`
- `metadata.json`

API 啟動與預測均會檢查以上檔案，缺任一檔將 fail-fast。

## Testing

- `flake8 agent.py`
- `flake8 src tests`
- `pytest -q`
