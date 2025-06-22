
# AGENTS.md
> **Scratch‑Card Prediction System – Agent‑Level Design**  
> **目標**：在「任意大小盤面」中以 *最短延遲* 取得 *最高準確* 的隱藏數字機率。

本文件說明 `app.py`, `main.py`, `analyzer.py`, `brain.py`, `modules.py` 之間的 _Agent_ 切分與責任。低耦合、高內聚的代理設計能讓你 **快速替換演算法**，同時維持 API/CLI 介面不變。

---

## 1. System Bird‑Eye

```
┌────────────┐ HTTP/JSON ┌──────────┐
│  Client    │──────────▶│ APIAgent │  (app.py / FastAPI)
└────────────┘           └──────────┘
                              │
                 RPC / Ray, in‑proc calls
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
* **CLI‑Agent** – `main.py`；提供批次/benchmark 用途。  
* **PredictionOrchestrator** – 核心協調層；負責封包向下分派、彙整機率。  
* **GridAgent** – 處理任意大小盤面 (`rows×cols ≤ 400`，實測可更大) 的解析、完整性檢查、向量化 mask。  
* **GenAgent** – 高效板面生成；`numpy` + `xxhash` 快取，確保重複格局 0 成本重用。  
* **SimAgent** – Monte‑Carlo / MCTS 批次模擬；Ray / Joblib 并行，支援自適應收斂。  
* **ScoreAgent** – 套用多重 heuristic/ML 模組，回傳每格得分向量。  
* **Formula Plugins** – `modules.py` 內之 `FORMULA_REGISTRY`；單一函式 = 一個策略，可熱插拔。  

---

## 2. Agent Contracts

| Agent | Public API | I/O 型別 | 關鍵特性 |
|-------|------------|----------|----------|
| **GridAgent** | `normalize(grid: np.ndarray) -> dict` | 任意 `-1/num` 2‑D 陣列 → dict{{known_mask, flat_values}} | 100 ns 內完成大小偵測 & 驗證 |
| **GenAgent**  | `sample(batch:int, ctx:GridCtx) -> NDArray[int16]` | 上下文 → `(batch, r, c)` | 向量化洗牌 + 熵權重；可 1e6 boards/s (4 CPU) |
| **ScoreAgent**| `score(board_batch) -> NDArray[float32]` | `(b,r,c)` → `(b,)` | 多公式合併；SIMD‑friendly |
| **SimAgent**  | `simulate(ctx, target:int, n_iter:int) -> ProbMap` | context, 目標, 迭代數 | 自適應收斂 (CV < 0.05)；Ray 任務粒度 500 |
| **PredictionOrchestrator** | `predict(grid, target, iterations) -> Result` | JSON‑like | 回傳 top‑k 與 full heat‑map |
| **APIAgent** | `/predict` | HTTP POST JSON | FastAPI + Pydantic；< 50 ms overhead |

### Return Schema (`Result`)
```jsonc
{
  "predictions": [
    {{ "row": 4, "col": 7, "probability": 24.3, "candidates": [71, 68, 74] }},
    …
  ],
  "full_probabilities": {
    "4,7": {{ "71": 0.243, "68": 0.051, "74": 0.032, … }},
    …
  }
}
```

---

## 3. Data Flow Steps

1. **Validate & Normalize** – APIAgent 取 JSON → `GridAgent.normalize`  
2. **Adaptive Iteration Planning** – Orchestrator 根據 `(rows×cols, known_ratio)` 決定 `iterations`, `batch_size`.  
3. **Parallel Simulation Loop**  

```python
while not converged:
    boards = GenAgent.sample(batch, ctx)
    scores = ScoreAgent.score(boards)
    SimAgent.update(scores)
```  

4. **Probability Aggregation** – Softmax on cumulative hit counts → cell x number map。  
5. **Top‑K Extraction** – 堆疊 (`heapq.nlargest`) 擷取信心最高格位。  
6. **Response Crafting** – Orchestrator 將 raw 機率轉百分比，API 返回。

---

## 4. Performance Checklist

| 類別 | 具體策略 |
|------|----------|
| 記憶體 | 以 `int16` 儲存格值；使用只讀 `np.broadcast_to` 免複製。|
| CPU | `numba`/SIMD‑friendly 函式，嚴禁 Python 級迴圈。|
| 並行 | `ray` 初始化在 CLI；在 API 端使用本地 thread‑pool，避免冷啟動成本。|
| 快取 | `@lru_cache` + `xxhash` 針對已知格生成 mask；重複呼叫 <1 µs。|
| 收斂 | 逐批計算 `σ/μ`，低於門檻即提前停止迭代。|

---

## 5. Adding / Replacing an Agent

1. **實作**：繼承對應 base class 或遵守同名函式簽章。  
2. **註冊**：  

```python
from modules import register_formula
@register_formula("MyCoolHeuristic", weight=0.15)
def my_formula(board: np.ndarray) -> float:
    …
```  

3. **單元測試**：放在 `tests/test_my_agent.py`，確保 < 1 ms/board。  
4. **Benchmark**：執行 `python -m benchmarks.speed --agent MyCoolHeuristic`.  

---

## 6. FAQ

| 問題 | 解答 |
|------|------|
| **Q**：盤面 30×30 會爆 RAM 嗎？ <br> **A**：不會，單板 ≈ 1.8 KB；1e5 boards ≈ 180 MB，可分批生成。 |
| **Q**：要提升準確率？ <br> **A**：1) 增加高品質公式 (多元啟發式)；2) 提升迭代數，必要時動態擴容 CPU。 |
| **Q**：速度不夠？ <br> **A**：檢查 `batch_size` 與 `ray` num_cpus；確保未落到序列化瓶頸。 |

---

> *最後更新：2025-06-22*  
> Maintainer: **橘子 (Research Master)**
