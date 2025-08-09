## 🧪 測試要求（Testing Standards）

每個 Agent 在提交前必須通過以下驗證：

### ✅ 程式碼靜態檢查

- `flake8 agent.py` 全綠燈（無錯誤或警告）

### ✅ 單元測試

測試檔案命名範例：`tests/test_agents/test_<agent_name>.py`

所有任務腳本完全實作 完整 不可簡化