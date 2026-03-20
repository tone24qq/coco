## 🧪 測試要求（Testing Standards）

每個 Agent 在提交前必須通過以下驗證：

### ✅ 程式碼靜態檢查

- `flake8 agent.py` 全綠燈（無錯誤或警告）

### ✅ 單元測試

測試檔案命名範例：`tests/test_agents/test_<agent_name>.py`

# Mainline Implementation Rules

## Definition of Done
只有同時滿足以下條件，才可宣稱完成：
1. 必須修改至少一個主線入口或主線共用核心：
   - src/train.py
   - src/predict.py
   - src/backtest.py
   - src/api.py
   - src/artifacts.py
   - src/build_features.py
   - src/runtime_scoring.py
   - src/modeling.py
2. 新功能必須真的被主流程呼叫到，不接受孤立 helper / 未接線模組。
3. 不接受只改 README / configs / tests / reports / response schema / metadata。
4. 不接受只新增檔案但主線資料流、模型流、runtime scoring 沒變。
5. 若未真正接入，必須明確回報「未完成」。

## Forbidden Shortcuts
以下情況一律視為未完成：
- 只補 README
- 只補 config
- 只補 tests
- 只補 API schema
- 只補 reports 輸出
- 只在 metadata 增加欄位
- 只做假 ablation / 假 consensus / 假 explain

## Required Final Report
最後回覆必須固定包含：
1. 本次判定的主線入口檔案
2. 真正修改的核心檔案
3. 每個核心檔案實際接入內容
4. 尚未完成項目
5. 實際執行命令
6. 實際通過的測試
7. 可驗證輸出檔案或日誌

## Required Workflow
1. 先列主線入口與最小修改檔案
2. 再實作
3. 再執行命令與測試
4. 沒證據不得宣稱完成