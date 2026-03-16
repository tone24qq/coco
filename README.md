# BingoBingo 訓練 / 推論（本機優先 SOP）

本專案目前的**正式預設流程**是 `cascade_v1`（Stage1→Stage2→Stage3 selector）。

- 本機直接執行 `python src/train_lgbm.py`，預設會走 `configs/train.yaml` 的 `pipeline.version`。
- 目前預設值已是 `cascade_v1`，因此你不需要額外切參數即可訓練最新版流程。

---

## 0) Pipeline precedence（非常重要）

### 訓練（`src/train_lgbm.py`）
1. `configs/train.yaml -> pipeline.version`（唯一來源）
2. 若為 `cascade_v1`：訓練 stage1/stage2 並輸出 `models/cascade_v1/*`
3. 訓練完成會更新 `models/strategy_config.json`，使預測預設指向 `cascade_v1_flow`

### 回測（`src/backtest.py`）
1. 讀 `configs/train.yaml`（資料窗、fold、pipeline 相關設定）
2. 讀 `configs/experiments.yaml`（實驗清單）
3. `cascade_v1_flow` 會走 cascade pipeline + selector 指標報表

### 預測（`src/predict.py` / `src/api.py`）
1. 若 `configs/predict.yaml -> pipeline.version != auto`：**強制使用 config 指定 pipeline**
2. 否則（`auto`）：使用 `models/strategy_config.json`（selected_strategy）
3. 若 strategy config 不完整，再回退 `models/metadata.json`
4. 若都缺，最後才用 defaults

> 建議本機日常使用：`predict.yaml` 維持 `auto`，讓它直接跟隨你最後一次訓練產物。

---

## 1) 本機最新版訓練 SOP（直接照打）

```bash
python src/prepare_data.py
python src/build_features.py
python src/train_lgbm.py
```

### 預期（最新版 cascade_v1）訓練產物

- `models/cascade_v1/stage1_model.cbm`
- `models/cascade_v1/stage1_feature_columns.json`
- `models/cascade_v1/stage2_model.cbm`
- `models/cascade_v1/stage2_feature_columns.json`
- `models/cascade_v1/stage3_input_schema.json`
- `models/cascade_v1/pipeline_metadata.json`
- `models/strategy_config.json`（`selected_strategy.version_id` 應為 `cascade_v1_flow`）
- `models/metadata.json`（含 `pipeline_artifacts.cascade_v1`）

---

## 2) 本機預測 SOP

### CLI
```bash
python src/predict.py
```

### API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 10000
```

- `POST /predict` 可用 `include_stage_details=true` 查看 stage debug（selector reason / stage keep count）
- `top3_numbers` = selector final top3
- `top3_no_selector` = stage2 raw top3
- `top10_numbers` = stage2 ranked top10

---

## 3) 本機回測 SOP（含 stagewise / selector uplift）

```bash
python src/backtest.py
```

### 重要報表
- `reports/experiment_registry.csv`
- `reports/experiment_per_fold_metrics.csv`
- `reports/history_bucket_report.csv`
- `reports/cascade_stagewise_report.json`

`cascade_stagewise_report.json` 會包含：
- stage1 recall@30 / retained count
- stage2 top10 / ndcg
- stage3 selector vs no-selector（exact/adj/strict_adj/distance uplift）
- history bucket selector uplift breakdown

---

## 4) 本機訓練完成檢查清單（快速驗收）

訓練後請逐條檢查：

1. `configs/train.yaml` 的 `pipeline.version` 是 `cascade_v1`
2. `models/cascade_v1/pipeline_metadata.json` 存在
3. `models/strategy_config.json` 的 `selected_strategy.stage_type == "cascade"`
4. `python src/predict.py` 輸出含 `strategy_version: cascade_v1_flow`
5. `python src/backtest.py` 後有 `reports/cascade_stagewise_report.json`

---

## 5) 一鍵本機驗證指令（開發者）

```bash
black --check .
isort --check-only .
flake8 .
flake8 agent.py
python -m py_compile $(git ls-files '*.py')
pytest -q
python scripts/normalize_gitignore.py
python scripts/pre_pr_checks.py
```

---

## 6) 目前已知限制

- Stage3 selector 為可解釋規則式組合打分（非學習式 selector）。
- 若手動把 `predict.yaml.pipeline.version` 設成非 `auto`，會覆蓋 strategy config，請確認是你要的行為。


---

## Local-first 資料流程（新版）

資料優先序：
1. local CSV (`data/raw`) primary
2. live current fetch（僅最新 5 分鐘即時增量）
3. hot/cold pages（冷熱、大小、單雙、跳號、分佈等技術面資料）

> 歷史資料完全來自本地資料夾；API request 期間不再抓官方 historical CSV。

### 建立 manifest 與 canonical dataset

```bash
python src/prepare_data.py
```

輸出：
- `data/raw/raw_manifest.json`
- `data/processed/bingo_draws_canonical.csv`
- `data/processed/bingo_draws_canonical.parquet`
- `reports/local_data_audit.json`

### 歷史回補（本地優先）

```bash
python scripts/backfill_history.py
```

> `POST /fetch/history-backfill` 已停用，避免 API request 期間觸發高記憶體歷史重建。

### OpenAPI 匯出

```bash
python scripts/export_openapi.py
```

### 新增 API 端點

- `POST /fetch/history-backfill`
- `POST /fetch/latest`
- `POST /fetch/consensus-check`
- `POST /features/rebuild`
- `POST /backtest/run`
- `GET /reports/source-consensus`
- `GET /reports/history-ablation`

### Render 部署

- 使用 `render.yaml` 或 `Procfile`
- 建議規格：>=4 vCPU / 8GB RAM


## 7) 產出物（Generated Artifacts）與 Git 策略

為避免 GitHub 單檔上限（100 MiB）與 repo 汙染，本專案將可重建產出物視為 generated artifacts。

### 不進 Git 的產出物

- `data/feature_store/`
- `data/processed/bingo_draws_canonical.csv`（已停用為預設輸出）
- `data/processed/bingo_draws_canonical.parquet`
- `data/processed/history_snapshot.parquet`
- `data/processed/history_snapshot_meta.json`
- `data/raw/raw_manifest.json`

### canonical 輸出策略

- 預設主格式：`parquet`
- 預設模式：`runtime`
  - 產出單一高效 parquet（執行期優先）
- 匯出模式：`export`
  - 若檔案超過 size guard（95 MiB），改為 deterministic sharded parquet dataset + `manifest.json`

### 指令

建立 canonical（執行期）

```bash
python src/prepare_data.py --artifact-mode runtime
```

建立 canonical（匯出/分享）

```bash
python src/prepare_data.py --artifact-mode export
```

建立 snapshot（執行期）

```bash
python scripts/build_history_snapshot.py --artifact-mode runtime
```

建立 snapshot（匯出/分享）

```bash
python scripts/build_history_snapshot.py --artifact-mode export
```

增量更新 canonical + snapshot

```bash
python scripts/update_history_snapshot.py --artifact-mode runtime
```
