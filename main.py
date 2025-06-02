# main.py
# 系統入口與 FastAPI 路由層。

# 來源：知識大典.txt – 防錯字典.txt – "PEP 8 代码风格指南" – "導入順序"
# 1. 標準庫導入
import asyncio
import logging
import os
import time # For performance logging
import uuid
from datetime import datetime, timezone, timedelta # timedelta for uptime
from typing import Any, Callable, Dict, List, Tuple, cast

# 2. 第三方庫導入
import httpx # For self_ping_task
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# 3. 本地應用/自定义模块导入
import analyzer
import brain

# 舊寫法 ❌ (dotenv loaded after some imports)
# load_dotenv()

# 新寫法 ✅ (PEP 8: imports grouped, then code)
load_dotenv()

# --- Global variables for app instance lifecycle ---
START_TIME: datetime | None = None
APP_INSTANCE_ID: str = f"app-instance-{uuid.uuid4()}"

# --- Configuration via Pydantic BaseSettings ---
class AppSettings(BaseSettings):
    app_name: str = Field(default="AI Scoring Service", env="APP_NAME")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    server_port: int = Field(default=8000, env="PORT") # Render typically sets PORT
    self_ping_base_url: str = Field(default=f"http://localhost:{os.getenv('PORT', 8000)}", env="SELF_PING_BASE_URL") # 新增：可配置的自身探測URL

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = AppSettings()

# --- Logging Setup ---
# 引用：知識大典.txt – 防錯字典.txt – "内建异常（Built-in Exceptions）及防范建议" – "KeyError" (間接防範：透過 get 方法及提供預設值)
# 引用：知識大典.txt – 除錯.txt – "Logging/日誌問題" – "記錄 request_id/trace_id" [cite: 537]
class RequestIdLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> Tuple[str, Any]:
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        current_request_id = kwargs['extra'].get(
            'request_id',
            self.extra.get('request_id', "NO_REQUEST_ID_PROVIDED")
        )
        kwargs['extra']['request_id'] = current_request_id
        return msg, kwargs

# 引用：知識大典.txt – 除錯.txt – "Logging/日誌問題" – "日誌格式化字串 (formatter) 中的變數必須與 LogRecord 屬性或 extra 參數中的鍵名完全一致" [cite: 537]
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - [%(request_id)s] - %(message)s"
)

_base_logger = logging.getLogger(settings.app_name)
logger = RequestIdLoggerAdapter(_base_logger, {'request_id': APP_INSTANCE_ID}) # Use APP_INSTANCE_ID as default for non-request logs


# --- Background Tasks ---
# 引用：知識大典.txt – 2024-2025知識全集.txt – "第5.1 AsyncIO 核心機制回顧"
async def keep_alive_task_main_app() -> None:
    """每60秒打印一次日誌以保持服務活躍，並監控系統資源（日誌本身即為監控）。"""
    while True:
        logger.info(f"💡 Keep-alive log for {APP_INSTANCE_ID}. Service active.", extra={"request_id": "keep_alive_log"})
        await asyncio.sleep(60)

# 引用：知識大典.txt – 2024-2025知識全集.txt – "第5.1 AsyncIO 核心機制回顧"
# 引用：知識大典.txt – 防錯字典.txt – "内建异常（Built-in Exceptions）及防范建议" (針對 httpx 可能拋出的異常)
async def self_ping_task() -> None:
    """每60秒對自身 /healthz 端點發起 HTTP GET 請求，以產生真實流量。"""
    healthz_url = f"{settings.self_ping_base_url.rstrip('/')}/healthz"
    logger.info(f"Self-ping task configured for URL: {healthz_url}", extra={"request_id": "self_ping_init"})
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                response = await client.get(healthz_url)
                response.raise_for_status() # Raises an exception for 4XX/5XX responses
                logger.info(f"🩺 Self-ping to {healthz_url} successful, status: {response.status_code}.",
                            extra={"request_id": "self_ping_success"})
            except httpx.RequestError as e:
                logger.error(f"🩺 Self-ping to {healthz_url} failed (RequestError): {e}",
                             exc_info=False,
                             extra={"request_id": "self_ping_failure"})
            except httpx.HTTPStatusError as e:
                logger.error(f"🩺 Self-ping to {healthz_url} failed (HTTPStatusError {e.response.status_code}): {e.response.text}",
                             exc_info=False,
                             extra={"request_id": "self_ping_http_failure"})
            except Exception as e:
                logger.error(f"🩺 Self-ping task encountered an unexpected error: {e}",
                             exc_info=True,
                             extra={"request_id": "self_ping_unexpected_error"})
            await asyncio.sleep(60)


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Module Scoring Service with keep-alive and self-ping tasks."
)

# --- FastAPI Event Handlers ---
ANALYZER_INSTANCE_CONFIG: analyzer.AnalyzerConfig # Type hint for global config

@app.on_event("startup")
async def startup_event() -> None:
    global ANALYZER_INSTANCE_CONFIG, START_TIME, APP_INSTANCE_ID
    START_TIME = datetime.now(timezone.utc)

    logger.info(f"🚀 Application Instance ID: {APP_INSTANCE_ID} starting up...", extra={"request_id": "startup"})
    logger.info(f"Log level set to: {settings.log_level}", extra={"request_id": "startup"})
    logger.info(f"Service will listen on port: {settings.server_port}", extra={"request_id": "startup"})
    logger.info(f"Self-ping base URL set to: {settings.self_ping_base_url}", extra={"request_id": "startup"})

    try:
        # 引用：An.txt (analyzer.py) - DEFAULT_ANALYZER_CONFIG 的使用
        ANALYZER_INSTANCE_CONFIG = analyzer.DEFAULT_ANALYZER_CONFIG
        analyzer.initialize_analyzer(config_override=ANALYZER_INSTANCE_CONFIG)
        logger.info("✅ Analyzer initialized successfully.", extra={"request_id": "startup"})
    except Exception as e:
        logger.error(f"🔥 CRITICAL: Failed to initialize analyzer during startup: {e}",
                     exc_info=True, extra={"request_id": "startup_analyzer_error"})
        # Consider app behavior: raise SystemExit("Critical component (Analyzer) failed to initialize.")

    asyncio.create_task(keep_alive_task_main_app())
    logger.info("✅ Keep-alive logging task started.", extra={"request_id": "startup"})

    asyncio.create_task(self_ping_task())
    logger.info("✅ Self-ping task started.", extra={"request_id": "startup"})

    logger.info("🏁 Application startup sequence complete.", extra={"request_id": "startup"})

@app.on_event("shutdown")
async def shutdown_event() -> None:
    global START_TIME, APP_INSTANCE_ID
    shutdown_time_utc = datetime.now(timezone.utc)
    uptime_message = "Uptime: Unknown (startup time not recorded)"
    if START_TIME:
        uptime = shutdown_time_utc - START_TIME
        uptime_message = f"Uptime: {str(uptime).split('.')[0]}"

    logger.info(f"🛑 Application Instance ID: {APP_INSTANCE_ID} shutting down. {uptime_message}",
                extra={"request_id": "shutdown", "app_instance_id": APP_INSTANCE_ID})
    # Add any other cleanup logic here (e.g., closing connections)
    logger.info("🏁 Application shutdown sequence complete.", extra={"request_id": "shutdown"})


# --- Request ID Middleware and Dependency ---
# 引用：知識大典.txt – 2024-2025知識全集.txt – "3.1.3 跨域、中間件與安全性設置" (概念相關：中間件處理請求)
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # 舊寫法 ❌ (request.state is not type-safe by default for custom attributes)
    # request.state.request_id = request_id

    # 新寫法 ✅ (Store in a more robust way or ensure state object exists and is typed if used heavily)
    # For this scope, we'll use a dictionary on state if state itself is a simple object.
    # However, FastAPI's Request.state is a State object, which behaves like a dict.
    request.state.request_id = request_id

    # Pass request_id to the logger adapter for this request's scope
    # This is a conceptual placement; actual loggers used in endpoints will need access.
    # The dependency `get_request_id` and the logger adapter's structure handle this.

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

async def get_request_id(request: Request) -> str:
    # 引用：知識大典.txt – 防錯字典.txt – "AttributeError" (防範：使用 getattr)
    return cast(str, getattr(request.state, "request_id", "unknown_request_id_in_dependency"))

# --- CORS Middleware ---
# 引用：知識大典.txt – 2024-2025知識全集.txt – "3.1.3 跨域、中間件與安全性設置"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD"], # Specify allowed methods
    allow_headers=["X-Request-ID", "Content-Type"], # Specify allowed headers
)

# --- Pydantic Models ---
class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="二維陣列代表當前盤面，-1表示空格")
    # 舊寫法 ❌ Optional[Dict[str, List[int]]]
    # 新寫法 ✅ PEP 604
    proposed_values: Dict[str, List[int]] | None = Field(default=None, description="提議的值")
    analyzer_config_override: analyzer.AnalyzerConfig | None = Field(default=None, description="可選：覆蓋預設的分析器設定")

    # 引用：知識大典.txt – 除錯.txt – "值錯誤 (ValueError)" (透過 validator 預防)
    # 引用：知識大典.txt – 除錯.txt – "型別錯誤 (TypeError)" (透過 Pydantic 的型別檢查預防)
    @field_validator('new_card') # 新寫法 ✅ (Pydantic V2)
    @classmethod
    def check_grid_not_empty_and_rectangular(cls, v: List[List[int]]) -> List[List[int]]:
        if not v:
            raise ValueError("new_card (grid) cannot be empty")
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("new_card must be a list of lists") # 簡化錯誤訊息
        if not v[0]: # Assuming if v is not empty, v[0] exists. If v can be [[]], this needs adjustment.
            raise ValueError("new_card rows cannot be empty if grid itself is not empty")
        row_len = len(v[0])
        if not all(len(row) == row_len for row in v):
            raise ValueError("All rows in new_card must have the same length (be rectangular)")
        for r_idx, row_data in enumerate(v):
            for c_idx, cell_val in enumerate(row_data):
                if not isinstance(cell_val, int):
                    raise ValueError(f"Cell ({r_idx},{c_idx}) must be an integer. Found: {cell_val} (type: {type(cell_val)})")
        return v

class Suggestion(BaseModel):
    coords: Tuple[int, int]
    confidence_score: float
    contributing_modules: Dict[str, float] | None = Field(default=None)

class AnalyzeResponse(BaseModel):
    request_id: str
    timestamp: str
    suggestions: List[Suggestion]
    grid_shape_analyzed: Tuple[int, int]
    message: str | None = Field(default=None)
    duration_ms: float

class ScoreModuleRequest(BaseModel):
    module_name: str = Field(...)
    grid_data: List[List[int]] = Field(..., description="二維陣列代表盤面")
    module_config_override: Dict[str, Any] | None = Field(default=None, description="可選：覆蓋該模組的預設Pydantic設定（JSON對象）")

    @field_validator('grid_data') # 新寫法 ✅ (Pydantic V2)
    @classmethod
    def check_score_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if not v:
            raise ValueError("grid_data cannot be empty")
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("grid_data must be a list of lists")
        if not v[0]:
            raise ValueError("grid_data rows cannot be empty")
        row_len = len(v[0])
        if not all(len(row) == row_len for row in v):
            raise ValueError("All rows in grid_data must have the same length")
        for r_idx, row_data in enumerate(v):
            for c_idx, cell_val in enumerate(row_data):
                if not isinstance(cell_val, int):
                    raise ValueError(f"Cell ({r_idx},{c_idx}) must be an integer. Found: {cell_val} (type: {type(cell_val)})")
        return v

class ScoreModuleResponse(BaseModel):
    request_id: str
    module_name: str
    scores_preview: List[List[float]] | None = Field(default=None, description="評分矩陣的預覽（最多5x5）")
    message: str
    duration_ms: float

# --- API Endpoints ---

# 引用：知識大典.txt – 防錯字典.txt – "PEP 8 代码风格指南" – "空行" (頂級函數間2空行)
@app.api_route("/", methods=["GET", "HEAD"], summary="Root Endpoint")
async def root(request: Request) -> Dict[str, str]:
    req_id = getattr(request.state, "request_id", "NO_ID_IN_ROOT")
    logger.info("📡 Root / endpoint hit.", extra={"request_id": req_id})
    return {"message": f"Service {settings.app_name} is alive. Instance: {APP_INSTANCE_ID}"}


@app.get("/healthz", summary="Basic Health Check (GET)")
async def healthz_get(request: Request) -> Dict[str, str]:
    req_id = getattr(request.state, "request_id", "NO_ID_IN_HEALTHZ_GET")
    logger.info("❤️ GET /healthz endpoint hit.", extra={"request_id": req_id})
    return {"status": "ok from main app"}


@app.api_route("/healthz", methods=["HEAD"], summary="Basic Health Check (HEAD)")
async def healthz_head(request: Request) -> Response:
    req_id = getattr(request.state, "request_id", "NO_ID_IN_HEALTHZ_HEAD")
    logger.info("❤️ HEAD /healthz endpoint hit.", extra={"request_id": req_id})
    return Response(status_code=200, media_type="text/plain", content="OK")


@app.api_route("/health", methods=["GET", "HEAD"], status_code=200, summary="Detailed Health Check")
async def health_detailed(request_id: str = Depends(get_request_id)) -> Dict[str, str]:
    logger.debug("💙 Detailed /health endpoint called.", extra={"request_id": request_id})
    uptime_str = "Unknown"
    if START_TIME:
        uptime = datetime.now(timezone.utc) - START_TIME
        uptime_str = str(uptime).split('.')[0]

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app_name": settings.app_name,
        "app_instance_id": APP_INSTANCE_ID,
        "uptime": uptime_str,
        "request_id": request_id
    }


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyze Grid for Suggestions")
async def analyze_route(
    data: AnalyzeRequest,
    request_id: str = Depends(get_request_id)
) -> AnalyzeResponse:
    start_time_perf = time.perf_counter() # 新寫法 ✅ (更精確的計時變數名)
    logger.info(f"Received /analyze request. Grid shape: {len(data.new_card)}x{len(data.new_card[0]) if data.new_card and data.new_card[0] else 'N/A'}",
                extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
    try:
        grid_np = np.array(data.new_card, dtype=int)
    except Exception as e: # 引用：知識大典.txt – 防錯字典.txt – "内建异常（Built-in Exceptions）及防范建议" (通用 Exception 捕獲)
        logger.error(f"Error converting new_card to NumPy array: {e}", exc_info=True, extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format in new_card: {str(e)}")

    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions: {grid_np.shape}", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=400, detail="Grid must be a 2D array and non-empty.")

    try:
        config_to_use = data.analyzer_config_override if data.analyzer_config_override else ANALYZER_INSTANCE_CONFIG
        # 引用：An.txt (analyzer.py) - analyze_grid 函數簽名
        suggestions_from_analyzer: List[Dict[str, Any]] = analyzer.analyze_grid(
            grid_np,
            request_id=request_id,
            analyzer_config_override=config_to_use
        )
        response_suggestions: List[Suggestion] = [
            Suggestion(**sug) for sug in suggestions_from_analyzer # type: ignore[arg-type] # Assuming sug matches Suggestion fields
        ]
        duration_ms = (time.perf_counter() - start_time_perf) * 1000
        logger.info(f"Successfully analyzed grid. Found {len(response_suggestions)} suggestions. Duration: {duration_ms:.2f}ms",
                    extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        return AnalyzeResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            suggestions=response_suggestions,
            grid_shape_analyzed=cast(Tuple[int, int], grid_np.shape),
            message="Analysis successful.",
            duration_ms=duration_ms
        )
    except HTTPException: # 引用：知識大典.txt – 防錯字典.txt – "BaseException" (建議捕獲更具體的異常，此處 HTTPException 是 FastAPI 特定處理)
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time_perf) * 1000
        logger.exception(f"An unexpected error occurred during analysis. Duration: {duration_ms:.2f}ms: {e}", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")


@app.post("/score", response_model=ScoreModuleResponse, summary="Score Grid with a Specific Module")
async def score_module_route(
    data: ScoreModuleRequest,
    request_id: str = Depends(get_request_id),
) -> ScoreModuleResponse:
    start_time_perf = time.perf_counter()
    logger.info(f"Received /score request for module: {data.module_name}", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})

    if data.module_name not in brain.REGISTERED_MODULES_BRAIN:
        logger.warning(f"Module '{data.module_name}' not found.", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=404, detail=f"Module '{data.module_name}' not found.")

    try:
        grid_np = np.array(data.grid_data, dtype=int)
    except Exception as e:
        logger.error(f"Error converting grid_data to NumPy array: {e}", exc_info=True, extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format: {str(e)}")

    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions for scoring: {grid_np.shape}", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=400, detail="Grid for scoring must be a 2D array and non-empty.")

    try:
        # 引用：An.txt - _get_module_specific_config_from_analyzer_config and DEFAULT_MODULE_CONFIGS logic
        default_config_for_module = brain.DEFAULT_MODULE_CONFIGS.get(data.module_name, brain.BaseModuleConfig())
        module_pydantic_config_class = type(default_config_for_module)
        final_module_config: brain.BaseModuleConfig

        if data.module_config_override:
            try:
                # 引用：建議.txt - Pydantic 配置的動態解析
                final_module_config = module_pydantic_config_class(**data.module_config_override)
                logger.info(f"Using overridden config for module {data.module_name}: {final_module_config.model_dump_json(indent=2)}",
                            extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
            except Exception as pydantic_error: # Catches Pydantic validation errors
                logger.error(f"Invalid module_config_override for {data.module_name}: {pydantic_error}", exc_info=True, extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
                raise HTTPException(status_code=400, detail=f"Invalid config override for module {data.module_name}: {pydantic_error}")
        else:
            final_module_config = default_config_for_module
            logger.info(f"Using default config for module {data.module_name}.", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})

        # 引用：An.txt - brain.get_module_score 函數簽名
        score_matrix: np.ndarray = brain.get_module_score(
            data.module_name,
            grid_np,
            config_override=final_module_config,
            request_id=request_id
        )

        preview: List[List[float]] | None = None
        if score_matrix.size > 0: # Check if score_matrix is not empty
            preview_rows = min(score_matrix.shape[0], 5)
            preview_cols = min(score_matrix.shape[1], 5)
            if preview_rows > 0 and preview_cols > 0 : # Ensure positive dimensions for slicing
                 preview = score_matrix[:preview_rows, :preview_cols].tolist()

        duration_ms = (time.perf_counter() - start_time_perf) * 1000
        logger.info(f"Successfully scored grid with module '{data.module_name}'. Duration: {duration_ms:.2f}ms", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})

        return ScoreModuleResponse(
            request_id=request_id,
            module_name=data.module_name,
            scores_preview=preview,
            message=f"Grid scored successfully with {data.module_name}.",
            duration_ms=duration_ms
        )
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time_perf) * 1000
        logger.exception(f"An unexpected error occurred while scoring with module {data.module_name}. Duration: {duration_ms:.2f}ms: {e}",
                         extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=500, detail=f"Internal server error during scoring: {str(e)}")

# --- Main execution for Uvicorn (primarily for local development) ---
if __name__ == "__main__":
    import uvicorn

    # Uvicorn log config customization
    # 引用：知識大典.txt – 除錯.txt – "Logging/日誌問題" (確保日誌格式化字串的正確性)
    log_config_uvicorn = uvicorn.config.LOGGING_CONFIG.copy() # Make a copy to modify
    log_config_uvicorn["formatters"]["default"]["fmt"] = (
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s" # Simplified for Uvicorn's own logs
    )
    log_config_uvicorn["formatters"]["access"]["fmt"] = ( # Standard access log format
        '%(asctime)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
    )

    # Use the application's logger for this startup message
    app_logger = logging.getLogger(settings.app_name)
    app_logger.info(
        f"🚀 Starting Uvicorn server directly from main.py on port {settings.server_port} for {settings.app_name}...",
        extra={"request_id": "main_direct_run"}
    )

    # 舊寫法 ❌ (Multiple uvicorn.run calls, one with reload=True unconditionally)
    # uvicorn.run("main:app", host="0.0.0.0", port=settings.server_port, log_config=log_config_uvicorn, reload=False)
    # uvicorn.run("main:app", host="0.0.0.0", port=settings.server_port, log_level=settings.log_level.lower(), reload=True, log_config=log_config_uvicorn)

    # 新寫法 ✅ (Single uvicorn.run, reload can be conditional for development)
    # For production, 'reload' should be False or omitted.
    # Render and other PaaS will use their own start commands, typically without reload.
    # This block is mainly for convenient local execution.
    RELOAD_APP = os.getenv("RELOAD_APP", "False").lower() in ("true", "1", "t")
    logger.info(f"Uvicorn reload mode: {RELOAD_APP}", extra={"request_id": "main_direct_run"})

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.server_port,
        log_config=log_config_uvicorn,
        log_level=settings.log_level.lower(),
        reload=RELOAD_APP
    )