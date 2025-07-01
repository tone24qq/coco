# AGENTS.md
> **Scratch-Card Prediction System – Agent-Level Design**  
> **目標**：在「任意大小盤面」中以 *最短延遲* 取得 *最高準確* 的隱藏數字機率。  
> Maintainer: **橘子 (Research Master)**  
> Last-Updated: **2025-06-25**

本文件說明 `app.py`, `main.py`, `analyzer.py`, `brain.py`, `modules.py` 之間的 _Agent_ 切分與責任。低耦合、高內聚的代理設計能讓你 **快速替換演算法**，同時維持 API/CLI 介面不變。
# AGENTS 使用與擴充規範（AGENTS.md）

本專案採模組化與可維護設計，所有預測邏輯基於啟發式演算法、歷史樣本統計、模擬推論。為保障主體邏輯清晰、穩定與可測試，請遵守以下規範：

---

## 🧾 允許修改的檔案範圍

預設允許修改以下五個主流程模組：

- `main.py`
- `app.py`
- `analyzer.py`
- `brain.py`
- `modules.py`

以及所有測試相關檔案：

- `tests/` 目錄下的檔案

---

## ✳️ 附加允許（必要時擴充）

若有明確功能需求，可**額外新增或修改最多 5 個 Python 檔案**（不含 tests），例如：

- `ml_predictor.py`、`grid_utils.py`、`agent_adapter.py` 等工具模組  
- 新增模組需保持與主流程邏輯一致，且命名清楚、結構單一

請避免破壞原始模組註冊與呼叫架構，如需大幅改動請先提議經審核通過。

---

## 🔒 套件依賴限制（Dependency Policy）

本專案依賴如下輕量套件：

- `numpy`, `scipy`, `xxhash`, `joblib`, `ray`
- `fastapi`, `uvicorn`, `pydantic`, `openpyxl`, `numba`

禁止新增下列大型套件：

- `pandas`, `scikit-learn`, `torch`, `transformers`, `keras`, `tensorflow` 等  
- **LightGBM** 可在外部訓練後將模型輸出為檔案，但不可於 repo 中安裝或使用其 `sklearn` wrapper 介面

---

## 📌 模型整合建議

如需整合機器學習邏輯：

1. 請於本地完成模型訓練與驗證，產出純檔案模型（例如 .txt/.bin）
2. 於允許範圍內實作 `@register_formula("ml_model")` 並將推論邏輯包裝進內部模組
3. 禁止在 repo 中引入外部 ML framework 或自動訓練流程

---

## 🧪 測試要求

所有新增模組必須提供對應測試，並保證：

- 不引入外部依賴或隱含副作用
- 通過現有 CI 測試與 lint 檢查
- 維持主流程的穩定性與介面一致性

---

如需進一步擴充範圍或討論技術整合，請事前提出修改計畫供審核。
---

## 1. System Bird-Eye

```
┌────────────┐ HTTP/JSON ┌──────────┐
│  Client    │──────────▶│ APIAgent │  (app.py / FastAPI)
└────────────┘           └──────────┘
                              │
                 RPC / Ray, in-proc calls
                              │
                   ┌───────────────────────┐
                   │ PredictionOrchestrator│  (analyzer.py)
                   └───────────────────────┘
                              │
          ┌───────────────────┼──────────────────┐
          │                   │                  │
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │ GridAgent  │     │ SimAgent   │     │ ScoreAgent │
   │ (brain.py) │     │ (analyzer) │     │ (modules)  │
   └────────────┘     └────────────┘     └────────────┘
          │                   │                  │
          └──────────┬────────┘                  │
                     ▼                           │
               ┌────────────┐                    │
               │ GenAgent   │ <─Formula plugins──┘
               │ (modules)  │
               └────────────┘
```

* **APIAgent** – FastAPI 入口；HTTP validation、exception mapping、CORS。  
* **CLI-Agent** – `main.py`；提供批次/benchmark 用途。  
* **PredictionOrchestrator** – 核心協調層；負責封包向下分派、彙整機率。  
* **GridAgent** – 處理任意大小盤面 (`rows×cols ≤ 400`，實測可更大) 的解析、完整性檢查、向量化 mask。  
* **GenAgent** – 高效板面生成；`numpy` + `xxhash` 快取，確保重複格局 0 成本重用。  
* **SimAgent** – Monte-Carlo / MCTS 批次模擬；Ray / Joblib 並行，支援自適應收斂。  
* **ScoreAgent** – 套用多重 heuristic/ML 模組，回傳每格得分向量。  
* **Formula Plugins** – `modules.py` 之 `FORMULA_REGISTRY`；單一函式 = 一個策略，可熱插拔。  
號碼不重複 號碼1-N(範圍依據行列大小數量）
---

## 2. Agent Contracts

| Agent | Public API | I/O 型別 | 關鍵特性 |
|-------|------------|----------|----------|
| **GridAgent** | `normalize(grid: np.ndarray) -> dict` | 任意 `-1/num` 2-D 陣列 → dict{{known_mask, flat_values}} | 100 ns 內完成大小偵測 & 驗證 |
| **GenAgent**  | `sample(batch:int, ctx:GridCtx) -> NDArray[int16]` | 上下文 → `(batch, r, c)` | 向量化洗牌 + 熵權重；可 1e6 boards/s (4 CPU) |
| **ScoreAgent**| `score(board_batch) -> NDArray[float32]` | `(b,r,c)` → `(b,)` | 多公式合併；SIMD-friendly |
| **SimAgent**  | `simulate(ctx, target:int, n_iter:int) -> ProbMap` | context, 目標, 迭代數 | 自適應收斂 (CV < 0.05)；Ray 任務粒度 500 |
| **PredictionOrchestrator** | `predict(grid, target, iterations) -> Result` | JSON-like | 回傳 top-k 與 full heat-map |
| **APIAgent** | `/predict` | HTTP POST JSON | FastAPI + Pydantic；< 50 ms overhead |

### Return Schema (`Result`)
```jsonc
{
  "predictions": [
    { "row": 4, "col": 7, "probability": 24.3, "candidates": [71, 68, 74] },
    …
  ],
  "full_probabilities": {
    "4,7": { "71": 0.243, "68": 0.051, "74": 0.032, … },
    …
  }
}
```

---

## 3. Data Flow Steps

1. **Validate & Normalize** – APIAgent 取 JSON → `GridAgent.normalize`  
2. **Adaptive Iteration Planning** – Orchestrator 根據 `(rows×cols, known_ratio)` 決定 `iterations`, `batch_size`。  
3. **Parallel Simulation Loop**

```python
while not converged:
    boards = GenAgent.sample(batch, ctx)
    scores = ScoreAgent.score(boards)
    SimAgent.update(scores)
```  

4. **Probability Aggregation** – Softmax on cumulative hit counts → cell x number map  
5. **Top-K Extraction** – 堆疊 (`heapq.nlargest`) 擷取信心最高格位  
6. **Response Crafting** – Orchestrator 將 raw 機率轉百分比，API 返回  

---

## 4. API Quick Reference （for Agents & Scripts）

🌟 **Scratch-Card Prediction API – Quick Reference**  
POST `https://coco-3clu.onrender.com/predict` (Content-Type: application/json)

```
┌─ Request JSON ──────────────────────────────────────────┐
│ grid_size      str   *必填*  盤面列×行，例如 "8x10"       │
│ board          2-D   可選   已知=實際號碼，未知=-1        │
│ target         int   可選   想預測的號碼；省略→全-1機率  │
│ iterations     int   可選   Monte-Carlo 次數，預設 5000   │
│ enable_legacy  bool  可選   true=載 Q5-Q10 舊模組         │
│ modules        list  可選   例 ["Q1","Q2","M1"]           │
└─────────────────────────────────────────────────────────┘
```

### 最小請求
```json
{ "grid_size": "8x10", "target": 71 }
```

### 完整請求
```json
{
  "grid_size": "4x5",
  "board": [[1,2,3,4,5],[6,7,8,9,10],[11,12,-1,14,15],[16,17,18,19,20]],
  "target": 13,
  "iterations": 10000,
  "enable_legacy": true,
  "modules": ["Q1","Q2","Q3","Q4","M1","R3"]
}
```

### Response (200)
```jsonc
{
  "board_shape": [8,10],
  "modules_used": ["Q1","Q2","Q3","Q4"],
  "probabilities": [[0.012,0.010,0.055,…], …],
  "meta": { "iterations":5000, "legacy":false, "elapsed_ms":146 }
}
```

#### 常見踩雷
| 錯誤 | 原因 / 解法 |
|------|------------|
| 500 missing-arg `target` | composite 模組呼叫子模組沒傳 `target` |
| inhomogeneous shape | 有模組回傳 1-D / scalar；需 `reshape` 成 `grid.shape` |
| 過慢 | 降 `iterations` 或 `enable_legacy=false` |

**TL;DR**  
1. 未知格用 **-1**；其他格 1…N 唯一整數  
2. 模組輸出 shape 必＝盤面 shape  
3. 快 → 關 legacy；準 → 開 legacy

---

## 5. Performance Checklist

| 類別 | 具體策略 |
|------|----------|
| 記憶體 | `int16` 儲存格值；`np.broadcast_to` 免複製 |
| CPU | `numba` / SIMD；嚴禁 Python 級迴圈 |
| 並行 | Ray + 本地 thread-pool；冷啟動預載 |
| 快取 | `@lru_cache` + `xxhash` 針對已知格 mask |
| 收斂 | 批次計算 CV；低於閾值提前停止 |

---

## 6. Adding / Replacing an Agent

1. **實作**：遵守現有函式簽章或繼承 base class  
2. **註冊**：  
```python
from modules import register_formula

@register_formula("MyCoolHeuristic", weight=0.15)
def my_formula(board: np.ndarray, *, target=None, **kw) -> np.ndarray:
    …
```  

3. **測試**：`tests/test_my_agent.py`，< 1 ms/board  
4. **Benchmark**：`python -m benchmarks.speed --agent MyCoolHeuristic`

---

## 7. FAQ

| 問題 | 解答 |
|------|------|
| 盤面 30×30 會爆 RAM 嗎？ | 不會，單板≈1.8 KB；1e5 boards≈180 MB，可分批 |
| 如何提升準確率？ | 增加高品質公式 + 提高迭代數 |
| 速度不夠？ | 調整 `batch_size`、`ray` num_cpus，避序列化瓶頸 |

---
#8
--- AGENTS.md
+++ AGENTS.md
@@ ## 7. FAQ
 ## 8. Enforcing Unit Tests
-All code submissions must include unit tests and pass CI before merge.
+All code submissions must include unit tests under `tests/` and pass CI:
+
+- **tests/** directory with files named `test_*.py`
+- **CI command**:  
+  ```bash
+  pytest --maxfail=1 --disable-warnings -q
+  ```
+
+Failure to include passing tests will block PR merges and deployments.
## 9. Heuristic Parameters (Q11 & Q12)

> 本節集中列出「尾數親和 Q11」與「等差序列 Q12」的可調常數，  
> 並示範如何在 **不改程式碼** 的情況下做 A/B 測試或熱調整。

| 模組 | 參數 | 預設值 | 型別 | 覆寫方式 | 說明 |
|------|------|--------|------|-----------|------|
| **Q11 – Global Digit Affinity** | `ALPHA` | `0.5` | float | `ENV: Q11_ALPHA` | 距離權重 w(d)=1 / (1 + α·d) 的 α |
| | `P` | `1` | int | `ENV: Q11_P` | 距離權重冪次 `d^p`；1 = 曼哈頓，2 = 歐氏 |
| | `SIM_TAIL` | `1.0` | float | 修改 `brain.DIGIT_SIM` | 尾數相同的相似度 |
| | `SIM_±10` | `0.7` | float | 同上 | 差 10/20 的相似度 |
| | `SIM_COMPLEMENT` | `0.4` | float | 同上 | 首尾互補（例 1↔31）相似度 |
| **Q12 – Arithmetic Progression** | `GAP_FUNC` | `1/(1+gap)` | str/λ | `ENV: Q12_GAP` (`"linear"`, `"inv_sq"`) | gap = max(|dr|,|dc|) 的權重函數 |
| | `DIRS` | 8 方向 | list(tuple) | 修改 `brain.Q12_DIRS` | `(dr, dc)` 方向集；可排除騎士步 |

### 覆寫範例

```bash
# 放在 docker-compose / Render Secret / GitHub Actions
export Q11_ALPHA=0.3
export Q11_P=2
export Q12_GAP="inv_sq"
效能備註
	•	Kernel Cache：Q11 使用 lru_cache(maxsize=32) 依 (H,W,α,p) 快取距離核
	•	卷積實作：盤面 ≤ 20×20 採 direct convolve2d；>20×20 自動改 fftconvolve
	•	向量化：Q12 先一次性 np.roll 產生 8 個 shifted view，再張量化驗證 AP

⸻

10. Parameter-Tuning Workflow
1.	建立 grid-search.yaml 定義 α、p、gap_func 搜尋空間
2.	執行python tools/tune_params.py --conf grid-search.yaml
3.	最佳組合自動寫回 .env.q11q12，CI 報表附帶 Hit-Rate / Latency
	4.	人工複核後，將 .env.q11q12 內容複製到 Render Secret 或 Production ENV

注意：任何參數覆寫都需同步更新 tests/test_q11_q12.py 的 fixture，
以確保回歸測試與 Production 行為一致。

# Scratch-Card Prediction System

This project provides a FastAPI service and CLI tool for predicting hidden numbers on scratch cards. The system uses modular heuristics and Monte-Carlo simulation to estimate probabilities.

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional, for development
```

## Running Tests

Execute all unit tests with:

```bash
pytest -q
```

## Indexing Convention

All API responses and configuration parameters use **1-based** row/column
indices. Internally, algorithms still operate with NumPy's 0-based indexing. For
example, a prediction for the top-left cell will be returned as `(row=1, col=1)`.

## Continuous Integration

GitHub Actions run linting and the test suite on every push and pull request. Slow tests are executed separately on a scheduled or manual trigger.

## Deployment on Render

To reduce build time when manually deploying to Render:

1. Switch your Render service to **Docker deploy** and use the provided `Dockerfile`.
   Because dependencies are installed in an earlier layer, they will be reused as long as `requirements.txt` remains unchanged.
2. Enable **Build Cache** under **Settings → Build & Deploy** and add the following path:

```
~/.cache/pip
```

Render will restore this cache before each build and save it afterwards, avoiding repeated downloads.

## Reliability & Accuracy Evaluation

This repository includes a suite of stress tests to verify the **actual accuracy and reliability** of prediction models.

### Key Components

- `tests/reliability_utils.py`  
  Core testing logic. Supports:
  - `run_infinite_test()` — Infinite simulation to observe accuracy drift.
  - `run_until_converged()` — Batches simulation until confidence interval < δ.

- `tests/test_reliable_accuracy.py`  
  Runs 1000 randomized trials for several board sizes. Useful for baseline comparison.

- `tests/test_accuracy_converges.py`  
  Verifies that the accuracy converges to a statistically valid estimate.

- `tests/reliable_accuracy_test_suite.py`  
  Lightweight wrapper for one-off accuracy testing.

### Example Usage

Run 1000 randomized trials to estimate accuracy:

```bash
python -m tests.reliable_accuracy_test_suite
```

Run stress test with live accuracy output:

```bash
python -m tests.reliability_utils run_infinite_test
```

Run convergence validation (used in CI):

```bash
pytest tests/test_accuracy_converges.py
```


