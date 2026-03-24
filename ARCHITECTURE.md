# ARCHITECTURE / 技術規格

## 1) Runtime 預測主線
`/predict` 執行順序：
1. fetch latest（multi-source, timeout/retries/backoff）
2. normalize latest（`issue`, `draw_time`, `n1..n20`）
3. merge with local history（dedupe by issue, conflict fail-fast）
4. time-sync validation（若 `latest_known_issue` 落後 fetched latest issue，直接 fail-fast）
5. build rank windows（僅使用 target issue 之前資料）
6. load transformer artifact + infer
7. top20 + top3 diversity post-ranking

## 2) Storage & I/O
- 主儲存：Parquet
  - `history_runtime.parquet`
  - `scores.parquet`
- CSV 相容入口保留：
  - `history_runtime.csv`
  - `scores.csv`
- local history 載入策略：若 `history_processed.parquet` 存在，優先讀 Parquet，否則讀 CSV。
- cache 為可控策略，不要求全量資料常駐記憶體。

## 3) Top3 Diversity + Relaxation
- strict constraints:
  - 尾數去重（3 個尾數全異）
  - 需跨 1-40 與 41-80
  - 避免相鄰號碼
- 若 strict 無解，啟用 constraint relaxation，改採最佳化組合。
- 輸出 `diversity_relaxed: true/false` metadata。

## 4) Tensor & Attention Contract
- raw candidate tensor：`[candidate=80, feature_dim]`
- model input tensor：`[candidate=80, feature_dim]`（feature_dim 可擴展，不寫死 reshape）
- encoder attention axis：candidate-to-candidate self-attention
- output score tensor：`[candidate=80]`

## 5) Artifact Contract + Drift Metadata
- runtime metadata (`metadata.json`):
  - `artifact_version`
  - `model_version`
  - `feature_version`
  - `history_artifact`
  - `score_artifact`
- transformer metadata (`transformer_metadata.json`):
  - `trained_up_to_issue`
  - `baseline_metrics`
  - `feature_version`
  - `required_input_schema`
- inference response drift metadata:
  - `trained_up_to_issue`
  - `baseline_metrics`
  - `feature_version`
  - `expected_input_schema`
