# 1-80 Candidate Ranking System

目前主線為：

1. `/predict` 執行時自動抓最新期數（multi-source + retries + timeout）
2. 正規化最新資料後與本地 history 合併（issue dedupe / conflict fail-fast）
3. 預測前做 time-sync validation（若 latest_known_issue 落後真實最新期數直接 fail-fast）
4. 建立最近視窗特徵（僅使用過去資料）
5. 載入 Small Transformer Encoder 模型輸出 1..80 ranking score
6. 回傳完整 `scores` + deterministic `top20` + diversity rerank `top3`

## Deploy contract（保留）

- `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
- `bash scripts/build_deploy_bundle.sh deploy_bundle`

## Runtime outputs

`src.runtime_history` 會產出（Parquet 優先 + CSV 相容）：

- `data/runtime_history/metadata.json`
- `data/runtime_history/transformer_metadata.json`
- `data/runtime_history/transformer_model.npz`
- `data/runtime_history/history_runtime.parquet`
- `data/runtime_history/scores.parquet`
- `data/runtime_history/history_runtime.csv`
- `data/runtime_history/scores.csv`

## API

- `GET /healthz`
- `GET /predict`
  - `latest_known_issue`
  - `target_issue`
  - `model_version`
  - `feature_version`
  - `data_source`
  - `fetch_attempts`
  - `scores`
  - `top20`
  - `top3`
  - `diversity_relaxed`
  - `drift_metadata`

分數是 **ranking score**，不是 calibrated probability。

詳細 tensor/attention 與 drift 規格請見 `ARCHITECTURE.md`。
