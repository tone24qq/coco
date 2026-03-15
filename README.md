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
```

---

## 6) 目前已知限制

- Stage3 selector 為可解釋規則式組合打分（非學習式 selector）。
- 若手動把 `predict.yaml.pipeline.version` 設成非 `auto`，會覆蓋 strategy config，請確認是你要的行為。
