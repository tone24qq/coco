# 1-80 Candidate Ranking System

本專案目前主線提供兩個正式入口：

- **Artifact 建置入口**：
  `python -m src.runtime_history --input data/processed/history_processed.csv --output data/runtime_history`
- **推論 API 入口**：`app.py`（FastAPI，提供 `/healthz` 與 `/predict`）

> 文件內容以 repo 內可執行程式碼為準，不使用不存在的 `train.py` 或其他假入口。

## Runtime artifact

`src.runtime_history` 會輸出：

- `data/runtime_history/metadata.json`
- `data/runtime_history/scores.csv`（完整 1..80 score chain）
- `data/runtime_history/history_runtime.csv`

artifact schema/version 不符時，inference 會直接報錯（fail-fast）。

## API

- `GET /healthz`: 健康檢查
- `GET /predict`: 呼叫 `src.inference`，回傳：
  - `scores`（完整 1..80 score chain）
  - `top20`（deterministic 排序）
  - `top3`（依 diversity post-ranking 規則）

分數是 ranking score，不是 calibrated probability。
