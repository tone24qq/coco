# Local Training Pipeline (Real-First + Synthetic + Holdout)

## 1) 從真盤建庫
```bash
python scripts/build_real_board_corpus.py \
  --output data/full_boards/full_board_corpus.jsonl \
  --audit reports/full_board_corpus_audit.json
```

## 2) 生成 synthetic profile + boards
```bash
python scripts/fit_real_board_generator.py \
  --real-corpus data/full_boards/full_board_corpus.jsonl \
  --output artifacts/synthetic_generator_profile.json

python scripts/generate_synthetic_boards.py \
  --real-corpus data/full_boards/full_board_corpus.jsonl \
  --profile artifacts/synthetic_generator_profile.json \
  --output data/full_boards/synthetic_board_corpus.jsonl \
  --per-real 12
```

## 3) build masking ranking dataset
```bash
python scripts/build_masked_ranking_dataset.py \
  --real-corpus data/full_boards/full_board_corpus.jsonl \
  --synthetic-corpus data/full_boards/synthetic_board_corpus.jsonl \
  --mask-ratios 0.1,0.2,0.3,0.5 \
  --masks-per-ratio 2 \
  --output data/ranking/ranking_dataset.parquet \
  --feature-schema artifacts/feature_schema.json
```

> 若資料量過大可加 `--shard-rows 2000000` 產生 shard+manifest。

## 4) 本地訓練
```bash
python scripts/train_local_ranker.py \
  --train-real-path data/ranking/train_real.parquet \
  --train-synth-path data/ranking/train_synth.parquet \
  --holdout-real-path data/ranking/holdout_real.parquet \
  --device auto \
  --max-workers auto
```

輸出：
- `artifacts/main_ranker.pkl`
- `artifacts/main_ranker_meta.json`
- `reports/train_local_ranker_report.json`

## 5) 跑 real holdout backtest
```bash
python scripts/run_real_holdout_backtest.py \
  --train-real data/ranking/train_real.parquet \
  --train-synth data/ranking/train_synth.parquet \
  --holdout-real data/ranking/holdout_real.parquet \
  --output reports/real_holdout_backtest_summary.json
```

## 6) 推理啟用新模型
`configs/inference.yaml`:
```yaml
trained_ranker:
  enabled: true
  strict_missing_artifact: true
```

- 啟用時會走：`feasibility gate -> trained ranker -> (可選) existing reranker`。
- 若 `artifacts/main_ranker.pkl` 缺失且 `strict_missing_artifact=true`，推理會 fail-fast。
