# main_api.py
# coding: utf-8

from fastapi import FastAPI, HTTPException, Body, Request
from typing import List, Dict, Optional, Any
import logging
from pydantic import BaseModel

# 假設 analyzer.py 和 main.py (提供模組) 在同一個目錄或 Python 路徑中
from analyzer import Analyzer, InitializationError, InvalidInputError, ModuleError, ModuleNotFoundError, ModuleExecutionError, VisualizationError

# ==============================================================================
# TODO: 非常重要！請確認這裡匯入的是你包含 GM 模組的正確檔案
# 如果你的 GM 模組檔案不叫 main.py，請修改下面的 'main'
# 例如：import my_gm_logic_module as main_logic_module
import main as main_logic_module # 您的核心邏輯模組
# ==============================================================================


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("api_service") # API 服務的 logger


# --- FastAPI App Setup ---
app = FastAPI(
    title="智慧評分系統 API (Smart Scoring System API)",
    description="提供基於進階模組的盤面分析與評分建議 API 服務。",
    version="1.1.0" # 假設版本更新
)


# --- Analyzer Instance ---
analyzer_instance: Optional[Analyzer] = None
try:
    analyzer_instance = Analyzer(main_module=main_logic_module, default_top_n=3)
    logger.info("Analyzer instance created successfully for API, using logic module: %s", getattr(main_logic_module, '__name__', 'N/A'))
except InitializationError as e_init:
    logger.critical("CRITICAL_ERROR: Failed to initialize Analyzer for API at startup: %s", e_init, exc_info=True)
except Exception as e:
    logger.critical("CRITICAL_ERROR: Unexpected error during Analyzer initialization for API: %s", e, exc_info=True)


# --- API Request & Response Models (Pydantic) ---
class AnalysisRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[int] # 根據您的範例，這是一維列表，若為座標+值，需調整
    active_modules: Optional[List[str]] = None
    module_weights: Optional[Dict[str, float]] = None
    top_n: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    message: Optional[str] = None
    analyzer_status: str

# 您可能需要為 /analyze 的成功回應定義一個更精確的 Pydantic 模型
# class AnalyzeSuccessResponse(BaseModel):
#     # ... 根據 analyzer_instance.analyze_board 的返回結構定義 ...
#     pass


# --- API Endpoints ---
@app.get("/",
         summary="服務根目錄與基礎健康 Ping",
         response_model=Dict[str, str],
         tags=["Utilities"])
async def read_root():
    """提供服務的根路徑，主要用於平台健康檢查或返回一個簡單的歡迎/狀態訊息。"""
    logger.info("Root path / was accessed.")
    return {"message": "智慧評分系統 API 正常運行中！ (Smart Scoring System API is running!)"}

@app.post("/analyze",
          summary="分析盤面並取得建議",
          response_model=Dict[str, Any], # 建議替換為更精確的 AnalyzeSuccessResponse 模型
          tags=["Analysis"])
async def analyze_board_endpoint(request_body: AnalysisRequest = Body(...)):
    """
    接收盤面狀態和候選值，調用核心分析器進行分析。
    - **new_card**: 當前盤面狀態 (二維整數列表, -1 表示未開)。
    - **proposed_values**: 候選的填入值列表。
    - **active_modules**: (可選) 指定要使用的模組名稱列表。
    - **module_weights**: (可選) 指定各模組的權重字典。
    - **top_n**: (可選) 每個 proposed_value 要回傳的 Top-N 建議數量。
    """
    if analyzer_instance is None:
        logger.error("API_ERROR /analyze: Analyzer instance is not available (initialization failed).")
        raise HTTPException(status_code=503, detail="分析服務暫時不可用，核心分析組件初始化失敗。")

    logger.info("API_CALL /analyze: Received request with %d proposed values. First PV (if any): %s",
                len(request_body.proposed_values),
                request_body.proposed_values[0] if request_body.proposed_values else "N/A")
    try:
        result = analyzer_instance.analyze_board(
            new_card=request_body.new_card,
            proposed_values=request_body.proposed_values,
            active_modules=request_body.active_modules,
            module_weights=request_body.module_weights,
            top_n=request_body.top_n
        )
        logger.info("API_SUCCESS /analyze: Analysis successful for %d proposed values.", len(request_body.proposed_values))
        return result
    except InvalidInputError as e:
        logger.warning("API_VALIDATION_ERROR /analyze: Invalid input: %s - Request: %s", e, request_body.model_dump_json(indent=2, exclude_none=True))
        raise HTTPException(status_code=422, detail=f"輸入參數無效 (Invalid Input): {e}")
    except ModuleNotFoundError as e:
        logger.error("API_MODULE_ERROR /analyze: Module not found: %s - Request: %s", e, request_body.model_dump_json(indent=2, exclude_none=True), exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析模組配置錯誤 (Module Not Found): {e}")
    except ModuleExecutionError as e:
        logger.error("API_MODULE_ERROR /analyze: Module execution error: %s - Request: %s", e, request_body.model_dump_json(indent=2, exclude_none=True), exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析模組執行錯誤 (Module Execution Error): {e}")
    except ModuleError as e: # General ModuleError from analyzer.py
        logger.error("API_MODULE_ERROR /analyze: General module error: %s - Request: %s", e, request_body.model_dump_json(indent=2, exclude_none=True), exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析模組通用錯誤 (Module Error): {e}")
    except VisualizationError as e: # Assuming this is a specific error type
        logger.error("API_VISUALIZATION_ERROR /analyze: Visualization error: %s - Request: %s", e, request_body.model_dump_json(indent=2, exclude_none=True), exc_info=True)
        raise HTTPException(status_code=500, detail=f"視覺化生成錯誤 (Visualization Error): {e}")
    except Exception as e:
        logger.critical("API_UNEXPECTED_ERROR /analyze: Unexpected critical error: %s - Request: %s", e, request_body.model_dump_json(indent=2, exclude_none=True), exc_info=True)
        raise HTTPException(status_code=500, detail=f"伺服器內部發生未預期錯誤 (Unexpected Server Error): {type(e).__name__} - {e}")

@app.get("/health",
         summary="健康檢查端點",
         response_model=HealthResponse,
         tags=["Utilities"])
async def health_check():
    """
    提供簡單的健康狀態檢查，確認 Analyzer 核心實例是否已成功初始化。
    """
    if analyzer_instance is None:
        logger.warning("HEALTH_CHECK /health: Failed - Analyzer not initialized.")
        return HealthResponse(status="unhealthy", reason="Analyzer core component not initialized.", analyzer_status="Not Initialized")
    logger.info("HEALTH_CHECK /health: Successful - Analyzer is initialized.")
    return HealthResponse(status="ok", message="Analyzer API is running and Analyzer core is initialized.", analyzer_status="Initialized")

# --- Uvicorn direct run (通常用於本地開發測試) ---
# 在 Render 等平台部署時，通常會使用 Procfile 或平台指定的啟動命令，例如：
# uvicorn main_api:app --host 0.0.0.0 --port ${PORT} --workers 1
#
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server directly from main_api.py for local development on port 8000...")
#     uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

