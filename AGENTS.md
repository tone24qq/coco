# 🤖 AGENTS.md – Scratch-Card Prediction System – Agent-Level Design

## 📌 目標（Goal）

設計一組可擴充的預測模組（Agents），能在任意大小的刮刮樂盤面中，以「最短延遲」取得「隱藏數字的高機率位置」。支援模擬、熱力圖、樣本配對、鄰近結構等多種策略。

---

## 🧠 Agent 設計原則（Design Principles）

每個 Agent 模組需遵守以下原則：

1. **功能單一**：專注實作一種策略邏輯（如 Monte Carlo、樣本、熱力圖、鄰近數等）
2. **介面統一**：
   ```python
   def predict(board: np.ndarray, target: int, **kwargs) -> List[Dict[str, Any]]:
       ...
   ```
   - `board`: 2D numpy array（含 -1 表示空格）
   - `target`: 欲預測的目標數字
   - 回傳：`[{row: int, col: int, score: float}, ...]`
3. **不可修改原盤面**
4. **可被單元測試調用**
5. **能被主流程注入使用（如 main.py / brain.py）**

---

## 🧪 測試要求（Testing Standards）

每個 Agent 在提交前必須通過以下驗證：

### ✅ 程式碼靜態檢查

- `flake8 agent.py` 全綠燈（無錯誤或警告）

### ✅ 單元測試

測試檔案命名範例：`tests/test_agents/test_<agent_name>.py`  
必須包含以下測項：

```python
import numpy as np
from agents.random_agent import predict  # 假設你的 agent 是 random_agent.py

def test_random_agent_predict_on_10x12():
    rng = np.random.default_rng(42)
    rows, cols = 10, 12
    grid = rng.integers(1, 100, size=(rows, cols))
    blank_indices = rng.choice(rows * cols, size=rng.integers(15, 26), replace=False)
    for idx in blank_indices:
        r, c = divmod(idx, cols)
        grid[r, c] = -1
    non_blanks = np.argwhere(grid != -1)
    target_r, target_c = non_blanks[rng.integers(len(non_blanks))]
    target = grid[target_r, target_c]
    result = predict(grid.copy(), target=target)
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert isinstance(item, dict)
        assert "row" in item and "col" in item and "score" in item
```

---

## 📦 交付準則（Delivery Criteria）

| 項目             | 說明 |
|------------------|------|
| `agent.py`       | 核心模組 |
| `test_agent.py`  | 單元測試，路徑為 `tests/test_agents/` |
| `agent_demo.py`  | CLI 測試腳本，可直接執行看結果 |
| 文檔註解         | 所有函式需附 `docstring` 說明邏輯 |
| 設計說明         | PR 需附上設計意圖與參考資料 |

---

## 🔍 整合測試點（Integration Checkpoints）

1. ✅ 可被 `main.py` 正確呼叫
2. ✅ 輸出結構與 `fuse_predictions_with_heatmap` 相容
3. ✅ 可登錄於 `brain.py > REGISTERED_MODULES_BRAIN`

---

## 📘 命名規則（Naming Rules）

| 類別         | 命名範例                  |
|--------------|---------------------------|
| Agent 檔案    | `monte_carlo_agent.py`     |
| 測試檔案      | `test_monte_carlo_agent.py` |
| 註冊名稱      | `"monte_carlo"`            |
| 主函式名稱    | `predict()`               |

---

## 🔚 結語

> 你可以大膽設計，因為我們會徹底測試。  
> — 橘子
