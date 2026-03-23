# Agent Implementation Specs (System Blueprint)

此文件定義了模型的底層行為與特徵工程協議，任何模型迭代必須遵循以下規範：

### 1. 數據演進策略 (3K - 1M Scale)
- **3,000 筆：** 基準驗證期，重點在於 `Omission (遺漏值)` 特徵的有效性。
- **100,000 筆：** 加入 `Co-occurrence (共現)` 矩陣，啟動 Transformer Self-Attention。
- **1,000,000 筆：** 全量訓練，開啟 `Cyclic (週期性)` 特徵捕捉，並執行 Walk-forward 長期驗證。

### 2. 核心特徵邏輯 (The 4 Soul Features)
1. **Omission (遺漏回歸)：** 計算 `Current_Gap` 與 `Avg_Gap`，捕捉幾何分布下的反彈訊號。
2. **Co-occurrence (局部連動)：** 透過 Attention 矩陣計算 80 個候選項間的「吸引/排斥」權重。
3. **Momentum (動態動量)：** 5/20/100 局滾動頻率 Delta 值，識別號碼受熱趨勢。
4. **Cyclic (時間週期)：** 局數末位與開獎時段的 Positional Encoding。

### 3. 模型約束 (Constraints)
- **Architecture:** Encoder-only Transformer (Layers=3, d_model=128)。
- **File Size:** Hard limit < 100MB (僅儲存 State Dict)。
- **Output:** 必須保留 1-80 的完整分數鏈 (0.0 - 1.0)。

### 4. Top 3 去重演算法 (Diversity Protocol)
排序後提取 Top 20，針對 Top 3 執行以下篩選：
- **尾數去重：** 同一尾數 (n%10) 在 Top 3 中不得重複超過 1 個。
- **區間去重：** Top 3 必須儘量跨越 1-40 與 41-80 兩大區間。
- **鄰近去重：** 避免輸出如 (22, 23, 24) 這種連號組合。
