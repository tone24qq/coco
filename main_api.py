# main_api.py
from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Optional, Any
import logging

# 假設 analyzer.py 和 main.py (提供模組) 在同一個目錄或 Python 路徑中
from analyzer import Analyzer, InvalidInputError, ModuleError, VisualizationError # 從 analyzer.py 匯入 Analyzer 和相關錯誤類別
import main as main_logic_module # 這是你實際的 main.py，提供 GMs 模組

# --- FastAPI App Setup ---
app = FastAPI(
title="智慧評分系統 API",
description="提供盤面分析與評分建議的 API 服務。",
version="1.0.0"
)

# --- Logging Setup ---
# (可以從 analyzer.py 複製或重新定義，確保 API 層也有日誌)
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("api")


# --- Analyzer Instance ---
# 你需要確保 main_logic_module 是你實際包含 GMs 模組的 main.py
# 這裡假設 main_logic_module 符合 Analyzer 的期望
try:
analyzer_instance = Analyzer(main_module=main_logic_module, default_top_n=3)
logger.info("Analyzer instance created successfully for the API.")
except Exception as e:
logger.critical("Failed to initialize Analyzer for the API: %s", e, exc_info=True)
# 如果 Analyzer 初始化失敗，API 可能無法正常工作，可以考慮退出或設置一個標誌
analyzer_instance = None # 或者引發啟動錯誤

# --- API Request Body Model (using Pydantic from FastAPI) ---
class AnalysisRequest(BaseModel):
new_card: List[List[int]]
proposed_values: List[int]
active_modules: Optional[List[str]] = None
module_weights: Optional[Dict[str, float]] = None
top_n: Optional[int] = None

# --- API Endpoint ---
@app.post("/analyze", summary="分析盤面並取得建議", response_model=Dict[str, Any])
async def analyze_board_endpoint(request_body: AnalysisRequest = Body(...)):
"""
接收盤面狀態和候選值，調用核心分析器進行分析。

- **new_card**: 當前盤面狀態，二維整數列表，-1 表示未開。
- **proposed_values**: 候選的填入值列表。
- **active_modules**: (可選) 指定要使用的模組名稱列表。
- **module_weights**: (可選) 指定各模組的權重字典。
- **top_n**: (可選) 每個 proposed_value 要回傳的 Top-N 建議數量。
"""
if analyzer_instance is None:
logger.error("Analyze endpoint called but Analyzer instance is not available.")
raise HTTPException(status_code=503, detail="分析服務暫時不可用，初始化失敗。")

logger.info("API /analyze endpoint called with PVs: %s", request_body.proposed_values)
try:
result = analyzer_instance.analyze_board(
new_card=request_body.new_card,
proposed_values=request_body.proposed_values,
active_modules=request_body.active_modules,
module_weights=request_body.module_weights,
top_n=request_body.top_n
)
logger.info("Analysis successful for PVs: %s, returning results.", request_body.proposed_values)
return result
except InvalidInputError as e:
logger.warning("Invalid input for /analyze: %s - Request: %s", e, request_body.model_dump_json())
raise HTTPException(status_code=422, detail=f"輸入參數無效: {e}")
except ModuleError as e: # 包括 ModuleNotFoundError, ModuleExecutionError
logger.error("Module error during /analyze: %s - Request: %s", e, request_body.model_dump_json(), exc_info=True)
raise HTTPException(status_code=500, detail=f"模組執行錯誤: {e}")
except VisualizationError as e:
logger.error("Visualization error during /analyze: %s - Request: %s", e, request_body.model_dump_json(), exc_info=True)
# 即使視覺化失敗，如果分析結果仍在，可以考慮返回部分結果或特定錯誤碼
raise HTTPException(status_code=500, detail=f"視覺化生成錯誤: {e}")
except Exception as e: # 其他未預期錯誤
logger.critical("Unexpected critical error during /analyze: %s - Request: %s", e, request_body.model_dump_json(), exc_info=True)
raise HTTPException(status_code=500, detail=f"伺服器內部發生未預期錯誤: {type(e).__name__}")

@app.get("/health", summary="健康檢查端點")
async def health_check():
"""提供簡單的健康狀態檢查。"""
if analyzer_instance is None:
return {"status": "unhealthy", "reason": "Analyzer not initialized"}
return {"status": "ok", "message": "Analyzer API is running"}

# 如果你想直接運行這個 API 檔案 (例如 python main_api.py)
# 你可以添加 uvicorn.run，但通常在部署時會用 uvicorn 命令列工具
# import uvicorn
# if __name__ == "__main__":
# uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")