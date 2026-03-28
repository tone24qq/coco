# ARCHITECTURE / 技術規格

## Entry points
- train: `src.train_transformer`
- backtest: `src.backtest_transformer`
- cli predict: `src.predict`
- api predict: `app.py -> src.inference.predict`

## Tensor contract
- raw tensor: `[batch, 80, feature_dim]`
- model input tensor: `[batch, 80, d_model]`
- attention axis: candidate-to-candidate self-attention
- model output: `[batch, 80]` ranking logits

## Feature contract
- `feature_version = rank_window_v2`
- 固定 `feature_names` 順序（train/backtest/predict 共用）
- mismatch 一律 fail-fast

## Runtime flow
1. fetch latest (multi-source full scan + retries + source parser + consensus diagnostics)
2. normalize latest
3. merge with local history
4. time-sync validation
5. load runtime metadata + transformer metadata
6. tensor/feature/version/drift checks
7. torch deterministic inference
8. top20 + top3 diversity rerank (strict then relaxation)

## History merge invariants
- merged history 允許 issue 缺期（non-consecutive），但必須 strict increasing 且無 duplicate。
- local/latest 重疊 issue 若欄位不一致，立即 fail-fast（conflict）。
- gap / rolling / retrieval 現行以 observed rows 為計算單位，而非連續 issue 真實距離。

## Storage / artifact
- Parquet-first, CSV compatibility kept.
- runtime_history only builds/syncs artifacts; never retrains.
- metadata must include drift fields:
  - trained_up_to_issue
  - baseline_metrics
  - feature_version
  - expected_input_schema
  - feature_names
  - tensor_contract

## Stale policy
- `stale_issues = current_issue - trained_up_to_issue`
- if stale_issues > stale_threshold, still return result but `is_stale = true`


## Predict observability fields
- `source_latest_issues`: per-source latest observed issue
- `source_records_count`: per-source validated full records count
- `source_tail_count`: per-source latest consecutive tail size
- `selected_source_reason`: deterministic source selection rationale
- `selected_source_full_records_count`: selected source full records count
- `selected_source_tail_count`: selected source latest tail size
- `consensus_status`: `unanimous` / `partial` / `divergent`
- `max_observed_issue`: max latest issue observed across successful sources
- `source_consensus.conflicts`: same-issue number conflicts across sources
- `raw_scores`, `raw_top20`, `raw_top3`: direct model ranking outputs
- `final_top20`, `final_top3`: externally consumed final ranking view
- `rerank_applied`, `rerank_reason`: whether optional rerank changed final view
