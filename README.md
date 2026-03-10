# BingoBingo 訓練 / 推論分離專案

本專案已整理成 **本地訓練、雲端只載模預測** 的流程。

## 專案結構

```text
project/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ feature_store/
├─ models/
├─ reports/
├─ src/
│  ├─ prepare_data.py
│  ├─ build_features.py
│  ├─ train_lgbm.py
│  ├─ backtest.py
│  ├─ predict.py
│  ├─ api.py
│  └─ utils.py
├─ configs/
│  ├─ train.yaml
│  └─ predict.yaml
└─ README.md
```

## 1) 本地訓練指令

```bash
python src/prepare_data.py
python src/build_features.py
python src/train_lgbm.py
python src/backtest.py
```

### 流程說明
- `prepare_data.py`：讀取 `賓果賓果_2023~2026.csv`，清洗為 `issue, draw_date, numbers`。
- `build_features.py`：建立單期特徵（僅用當期以前資料），輸出特徵表與固定欄位清單。
- `train_lgbm.py`：訓練 CatBoost，輸出模型、metadata、特徵重要度。
- `backtest.py`：使用 `TimeSeriesSplit` 做 walk-forward 回測，輸出每折與總指標。

## 2) 上線預測指令

```bash
uvicorn src.api:app --host 0.0.0.0 --port 10000
```

API 端點：
- `GET /health`
- `GET /analysis`
- `POST /predict`

> API **只載入已訓練模型**，不會重新訓練。

## 3) 訓練輸出物

訓練後會產生：
- `models/catboost_top20.cbm`
- `models/feature_columns.json`
- `models/metadata.json`
- `reports/backtest_metrics.json`
- `reports/feature_importance.csv`
- `reports/walkforward_report.csv`

## 4) 重訓與替換模型

### 如何重訓
1. 更新根目錄歷史資料 CSV（或 `data/raw`）。
2. 重新執行四步訓練指令。
3. 新模型會覆蓋 `models/` 舊檔。

### 如何替換上線模型
- 將本地最新 `models/` 檔案部署到雲端相同路徑。
- 重啟 FastAPI 服務即可生效。

## 特徵一致性保證

- 訓練與推論都透過 `src/utils.py::build_issue_features` 與 `build_candidate_matrix`。
- 避免 train/inference feature mismatch。
- 回測使用時間序列切分，避免未來資料洩漏。
