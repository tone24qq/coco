# AGENTS.md
> **Scratch-Card Prediction System – Agent-Level Design**  
> **目標**：在「任意大小盤面」中以 *最短延遲* 取得 *最高準確* 的隱藏數字機率。  
> Maintainer: **橘子 (Research Master)**  
> Last-Updated: **2025-06-25**

本文件說明 `app.py`, `main.py`, `analyzer.py`, `brain.py`, `modules.py` 之間的 _Agent_ 切分與責任。低耦合、高內聚的代理設計能讓你 **快速替換演算法**，同時維持 API/CLI 介面不變。

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
