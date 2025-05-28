您提供的 main_api.py 檔案結構看起來非常清晰和專業！日誌設定、Analyzer 實例的初始化與錯誤處理、Pydantic 模型的運用以及 API 端點的設計都相當不錯。
針對我們之前討論的「服務在 Render 上因根路徑 / 回應 404 而關閉」的問題，您需要在這個 main_api.py 檔案中加入一個處理根路徑 / 的端點。
最佳的置入位置是在您定義 app = FastAPI(...) 之後，以及在您現有的 /analyze 和 /health 端點之前（或者與它們並列）。
以下是修改後的 main_api.py，我已經為您加入了處理根路徑 / 的 @app.get("/") 端點。我只添加了這個新端點，其餘部分保留了您的原始結構和邏輯：
# main_api.py
from fastapi import FastAPI, HTTPException, Body, Request # <--- 為根路徑的 request 參數加入 Request (如果需要)
from typing import List, Dict, Optional, Any # Union is used implicitly by Optional
import logging
from pydantic import BaseModel # Import BaseModel for request body

# 假設 analyzer.py 和 main.py (提供模組) 在同一個目錄或 Python 路徑中
from analyzer import Analyzer, InitializationError, InvalidInputError, ModuleError, ModuleNotFoundError, ModuleExecutionError, VisualizationError # 從 analyzer.py 匯入 Analyzer 和相關錯誤類別

# ==============================================================================
# TODO: 非常重要！請確認這裡匯入的是你包含 GM 模組的正確檔案
# 如果你的 GM 模組檔案不叫 main.py，請修改下面的 'main'
# 例如：import my_gm_logic as main_logic_module
import main as main_logic_module
# ==============================================================================


# --- Logging Setup ---
# 確保 API 層也有日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("api") # 給 API 相關的日誌一個獨立的 logger name


# --- FastAPI App Setup ---
app = FastAPI(
    title="智慧評分系統 API",
    description="提供盤面分析與評分建議的 API 服務。",
    version="1.0.0"
)


# --- Analyzer Instance ---
# 你需要確保 main_logic_module 是你實際包含 GMs 模組的 main.py
# 這裡假設 main_logic_module 符合 Analyzer 的期望
analyzer_instance: Optional[Analyzer] = None # 最好先定義一個預設值並註明類型
try:
    # 下面這行是第 30 行左右 (根據你之前的截圖)，已修正縮排
    analyzer_instance = Analyzer(main_module=main_logic_module, default_top_n=3)
    logger.info("Analyzer instance created successfully for the API using main_logic_module: %s", getattr(main_logic_module, '__name__', 'N/A'))
except InitializationError as e_init: # 更具體地捕獲 Analyzer 初始化錯誤
    logger.critical("CRITICAL: Failed to initialize Analyzer during API startup: %s", e_init, exc_info=True)
    # 在這種情況下，analyzer_instance 將保持為 None，API 端點會返回服務不可用
except Exception as e: # 捕獲其他可能的匯入或配置錯誤
    logger.critical("CRITICAL: An unexpected error occurred during Analyzer initialization for API: %s", e, exc_info=True)
    # analyzer_instance 保持為 None


# --- API Request Body Model (using Pydantic from FastAPI) ---
class AnalysisRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[int]
    active_modules: Optional[List[str]] = None
    module_weights: Optional[Dict[str, float]] = None
    top_n: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    reason: Optional[str] = None
    message: Optional[str] = None
    analyzer_status: str


# --- API Endpoints ---

# ============================================================
#  👇👇👇 新增的根路徑 "/" 處理函式 👇👇👇
# ============================================================
@app.get("/",
         summary="服務根目錄與基礎健康檢查",
         response_model=Dict[str, str], # 可以定義一個簡單的 Pydantic 模型，或直接用 Dict
         tags=["Utilities"])
async def read_root():
    """
    提供服務的根路徑，主要用於平台健康檢查或返回一個簡單的歡迎/狀態訊息。
    """
    logger.info("Root path / was accessed.")
    return {"message": "智慧評分系統 API 正常運行中！ (Smart Scoring System API is running and healthy!)"}
# ============================================================


@app.post("/analyze",
          summary="分析盤面並取得建議",
          response_model=Dict[str, Any], # 或者更精確的 Pydantic Response Model
          tags=["Analysis"])
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
        logger.error("API /analyze: Analyzer instance is not available (initialization failed).")
        raise HTTPException(status_code=503, detail="分析服務暫時不可用，核心組件初始化失敗。")

    logger.info("API /analyze endpoint called with %d PVs. First PV (if any): %s",
                len(request_body.proposed_values),
                request_body.proposed_values[0] if request_body.proposed_values else "N/A")
    try:
        # 調用 Analyzer 的核心方法
        result = analyzer_instance.analyze_board(
            new_card=request_body.new_card,
            proposed_values=request_body.proposed_values,
            active_modules=request_body.active_modules,
            module_weights=request_body.module_weights,
            top_n=request_body.top_n
        )
        logger.info("API /analyze: Analysis successful for %d PVs. Returning results.", len(request_body.proposed_values))
        return result
    except InvalidInputError as e:
        logger.warning("API /analyze: Invalid input: %s - Request: %s", e, request_body.model_dump_json(indent=2))
        raise HTTPException(status_code=422, detail=f"輸入參數無效 (Invalid Input): {e}")
    except ModuleNotFoundError as e: # 模組未找到
        logger.error("API /analyze: Module not found during analysis: %s - Request: %s", e, request_body.model_dump_json(indent=2), exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析模組配置錯誤 (Module Not Found): {e}")
    except ModuleExecutionError as e: # 模組執行錯誤
        logger.error("API /analyze: Module execution error during analysis: %s - Request: %s", e, request_body.model_dump_json(indent=2), exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析模組執行錯誤 (Module Execution Error): {e}")
    except ModuleError as e: # 其他 analyzer.py 定義的 ModuleError
        logger.error("API /analyze: General module error during analysis: %s - Request: %s", e, request_body.model_dump_json(indent=2), exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析模組通用錯誤 (Module Error): {e}")
    except VisualizationError as e:
        logger.error("API /analyze: Visualization error during analysis: %s - Request: %s", e, request_body.model_dump_json(indent=2), exc_info=True)
        # 即使視覺化失敗，如果分析結果仍在，可以考慮返回部分結果或特定錯誤碼
        # 但為了簡化，這裡也拋出 500 錯誤
        raise HTTPException(status_code=500, detail=f"視覺化生成錯誤 (Visualization Error): {e}")
    except Exception as e: # 其他所有未預期的錯誤
        logger.critical("API /analyze: Unexpected critical error during analysis: %s - Request: %s", e, request_body.model_dump_json(indent=2), exc_info=True)
        raise HTTPException(status_code=500, detail=f"伺服器內部發生未預期錯誤 (Unexpected Server Error): {type(e).__name__} - {e}")

@app.get("/health",
         summary="健康檢查端點",
         response_model=HealthResponse,
         tags=["Utilities"])
async def health_check():
    """
    提供簡單的健康狀態檢查。
    檢查 Analyzer 核心實例是否已成功初始化。
    """
    if analyzer_instance is None:
        logger.warning("/health: Health check failed - Analyzer not initialized.")
        return HealthResponse(status="unhealthy", reason="Analyzer core component not initialized.", analyzer_status="Not Initialized")
    logger.info("/health: Health check successful - Analyzer is initialized.")
    return HealthResponse(status="ok", message="Analyzer API is running and Analyzer core is initialized.", analyzer_status="Initialized")

# --- Uvicorn direct run (optional, for local testing) ---
# 通常在部署時，你會使用 Procfile 或類似的配置來由 Uvicorn 命令行工具啟動
# 例如：uvicorn main_api:app --host 0.0.0.0 --port 8000
#
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server directly from main_api.py for local development...")
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") # 可以用 "main_api:app" 或直接傳 app 物件

主要變更點：
 * 新增了 @app.get("/") 路由：
   @app.get("/",
         summary="服務根目錄與基礎健康檢查",
         response_model=Dict[str, str], 
         tags=["Utilities"])
async def read_root():
    """
    提供服務的根路徑，主要用於平台健康檢查或返回一個簡單的歡迎/狀態訊息。
    """
    logger.info("Root path / was accessed.")
    return {"message": "智慧評分系統 API 正常運行中！ (Smart Scoring System API is running and healthy!)"}

   * 我將它放在了 /analyze 和 /health 端點定義之前，這是一個常見的做法，但只要它在 app = FastAPI() 之後定義即可。
   * 它返回一個簡單的 JSON 回應和 200 OK 狀態碼 (FastAPI 預設)。
   * response_model=Dict[str, str] 是一個簡單的示例，您也可以為此定義一個更具體的 Pydantic 模型，如果需要的話。
   * 我也加入了 logger.info，這樣您可以在日誌中看到根路徑何時被訪問。
部署到 Render 時的提醒：
 * 確保您的 Render 服務的啟動命令是 uvicorn main_api:app --host 0.0.0.0 --port ${PORT} （或者根據您的需求調整 worker 數量等參數，例如 uvicorn main_api:app --host 0.0.0.0 --port ${PORT} --workers 1）。
 * 在 Render 的服務設定中，通常可以將健康檢查路徑 (Health Check Path) 保留為預設的 /。現在您的應用程式會對此路徑正確回應 200 OK。
將這個修改後的 main_api.py 部署到 Render 後，因為根路徑 / 現在會返回成功的 HTTP 狀態，您的服務應該不會再因為健康檢查失敗而被平台關閉了。
