# Log Usage Guide

本文件簡要列出常見的 logger 訊息與其用途，方便新進開發或實習生快速掌握排錯重點。

## API 相關（app.py）
- `Predict | size=%dx%d ...`：每次 /predict 呼叫時輸出，含主要參數，便於追蹤請求來源與設定。
- `No priors for %s...`：缺少對應尺寸的先驗機率檔案時觸發，代表系統會即時計算。
- `✅ Response ready`：預測流程完成並已產生回應。
- `Prediction failed`、`Heatmap failed`、`Fusion failed`：處理過程發生例外，需檢查堆疊訊息。
- `Starting API on port`、`Shutdown complete`：服務啟停紀錄。

## 分析模組（analyzer.py）
- `loaded %dx%d heatmap from`：載入全域熱力圖檔成功，確認快取是否可用。
- `Loaded %d sample boards...`：樣本盤面載入情形，判斷資料量。
- `匹配到%d张样本...`：預測時樣本比對結果，能評估模擬準確度。
- `simulate_full_board called`：Monte Carlo 模擬啟動，顯示迭代次數。
- `neighbor_lock` 系列訊息：鄰居鎖定流程的候選數與最終選擇。

## CLI（main.py）
- `Prediction results (strategy=%s)`：列出所使用的策略名稱。
- `Full probability maps computed` / `Heatmap saved`：執行 heatmap 功能的輸出結果。

如需更詳細的日誌說明，可直接查閱對應程式碼旁的中文註解。
