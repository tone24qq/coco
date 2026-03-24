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
1. fetch latest (multi-source failover + retries + source parser)
2. normalize latest
3. merge with local history
4. time-sync validation
5. load runtime metadata + transformer metadata
6. tensor/feature/version/drift checks
7. torch deterministic inference
8. top20 + top3 diversity rerank (strict then relaxation)

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
