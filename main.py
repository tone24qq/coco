# main.py
# 系統入口與 FastAPI 路由層。

import os
import logging
import uuid
from datetime import datetime, timezone, timedelta # timedelta for uptime
import time # For performance logging
import asyncio # For background tasks
import httpx # For self_ping_task

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Tuple, cast
from typing import Callable
from dotenv import load_dotenv

load_dotenv()

import analyzer
import brain

# --- Global variables for app instance lifecycle ---
START_TIME: datetime | None = None
APP_INSTANCE_ID: str = f"app-instance-{uuid.uuid4()}"

# --- Configuration via Pydantic BaseSettings ---
class AppSettings(BaseSettings):
    app_name: str = Field(default="AI Scoring Service", env="APP_NAME")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    server_port: int = Field(default=8000, env="PORT") # Render typically sets PORT

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = AppSettings()

# --- Logging Setup ---
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

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - [%(request_id)s] - %(message)s"
)

_base_logger = logging.getLogger(settings.app_name)
logger = RequestIdLoggerAdapter(_base_logger, {'request_id': APP_INSTANCE_ID}) # Use APP_INSTANCE_ID as default for non-request logs


# --- Background Tasks ---
async def keep_alive_task_main_app():
    """每60秒打印一次日誌以保持服務活躍，並監控系統資源（日誌本身即為監控）。"""
    while True:
        # "監控系統資源" 目前主要透過日誌表示活躍。可擴展加入 psutil 等真實資源監控。
        logger.info(f"💡 Keep-alive log for {APP_INSTANCE_ID}. Service active.", extra={"request_id": "keep_alive_log"})
        await asyncio.sleep(60)

async def self_ping_task():
    """每60秒對自身 /healthz 端點發起 HTTP GET 請求，以產生真實流量。"""
    # 注意：這裡的 URL 和 PORT 應與服務實際監聽的配置一致
    # 在 Render 環境中，PORT 通常由平台設定，settings.server_port 應能讀取到
    healthz_url = url = "https://coco-3clu.onrender.com/healthz"
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                response = await client.get(healthz_url)
                response.raise_for_status() # Raises an exception for 4XX/5XX responses
                logger.info(f"🩺 Self-ping to {healthz_url} successful, status: {response.status_code}.",
                            extra={"request_id": "self_ping_success"})
            except httpx.RequestError as e:
                logger.error(f"🩺 Self-ping to {healthz_url} failed (RequestError): {e}",
                             exc_info=False, # Avoid full stack trace for common ping errors
                             extra={"request_id": "self_ping_failure"})
            except httpx.HTTPStatusError as e:
                logger.error(f"🩺 Self-ping to {healthz_url} failed (HTTPStatusError {e.response.status_code}): {e}",
                             exc_info=False,
                             extra={"request_id": "self_ping_http_failure"})
            except Exception as e:
                logger.error(f"🩺 Self-ping task encountered an unexpected error: {e}",
                             exc_info=True, # Log full trace for unexpected errors
                             extra={"request_id": "self_ping_unexpected_error"})
            await asyncio.sleep(60)


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Module Scoring Service with keep-alive and self-ping tasks."
)

# --- FastAPI Event Handlers ---
ANALYZER_INSTANCE_CONFIG: analyzer.AnalyzerConfig

@app.on_event("startup")
async def startup_event():
    global ANALYZER_INSTANCE_CONFIG, START_TIME, APP_INSTANCE_ID
    START_TIME = datetime.now(timezone.utc)

    logger.info(f"🚀 Application Instance ID: {APP_INSTANCE_ID} starting up...", extra={"request_id": "startup"})
    logger.info(f"Log level set to: {settings.log_level}", extra={"request_id": "startup"})
    logger.info(f"Service will listen on port: {settings.server_port}", extra={"request_id": "startup"})

    try:
        ANALYZER_INSTANCE_CONFIG = analyzer.DEFAULT_ANALYZER_CONFIG # Or load from settings if implemented
        analyzer.initialize_analyzer(config_override=ANALYZER_INSTANCE_CONFIG)
        logger.info("✅ Analyzer initialized successfully.", extra={"request_id": "startup"})
    except Exception as e:
        logger.error(f"🔥 CRITICAL: Failed to initialize analyzer during startup: {e}",
                     exc_info=True, extra={"request_id": "startup_analyzer_error"})
        # Consider if app should proceed or exit if analyzer is critical

    # 啟動背景任務
    asyncio.create_task(keep_alive_task_main_app())
    logger.info("✅ Keep-alive logging task started.", extra={"request_id": "startup"})

    asyncio.create_task(self_ping_task())
    logger.info("✅ Self-ping task to /healthz started.", extra={"request_id": "startup"})

    logger.info("🏁 Application startup sequence complete.", extra={"request_id": "startup"})

@app.on_event("shutdown")
async def shutdown_event():
    global START_TIME, APP_INSTANCE_ID
    shutdown_time_utc = datetime.now(timezone.utc)
    uptime_message = "Uptime: Unknown (startup time not recorded)"
    if START_TIME:
        uptime = shutdown_time_utc - START_TIME
        uptime_message = f"Uptime: {str(uptime).split('.')[0]}" # Remove microseconds for cleaner log

    logger.info(f"🛑 Application Instance ID: {APP_INSTANCE_ID} shutting down. {uptime_message}",
                extra={"request_id": "shutdown", "app_instance_id": APP_INSTANCE_ID})
    # Add any other cleanup logic here (e.g., closing connections)
    logger.info("🏁 Application shutdown sequence complete.", extra={"request_id": "shutdown"})


# --- Request ID Middleware and Dependency ---
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

async def get_request_id(request: Request) -> str:
    return cast(str, getattr(request.state, "request_id", "unknown_request_id_in_dependency"))

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ( Definitions from previous corrected version ) ---
class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="二維陣列代表當前盤面，-1表示空格")
    proposed_values: Dict[str, List[int]] | None = Field(default=None, description="提議的值")
    analyzer_config_override: analyzer.AnalyzerConfig | None = Field(default=None, description="可選：覆蓋預設的分析器設定")
    @validator('new_card')
    def check_grid_not_empty_and_rectangular(cls, v: List[List[int]]) -> List[List[int]]:
        if not v: raise ValueError("new_card (grid) cannot be empty")
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("new_card must be a list of lists of integers")
        if not v[0]: raise ValueError("new_card rows cannot be empty if grid itself is not empty")
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
    @validator('grid_data')
    def check_score_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if not v: raise ValueError("grid_data cannot be empty")
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("grid_data must be a list of lists of integers")
        if not v[0]: raise ValueError("grid_data rows cannot be empty")
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

@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request):
    req_id = getattr(request.state, "request_id", "NO_ID_IN_ROOT")
    logger.info("📡 Root / endpoint hit.", extra={"request_id": req_id})
    return {"message": "Service is alive."}

@app.get("/healthz")
async def healthz_get(request: Request):
    req_id = getattr(request.state, "request_id", "NO_ID_IN_HEALTHZ_GET")
    logger.info("❤️ GET /healthz endpoint hit.", extra={"request_id": req_id})
    return {"status": "ok from main app"}

@app.api_route("/healthz", methods=["HEAD"])
async def healthz_head(request: Request):
    req_id = getattr(request.state, "request_id", "NO_ID_IN_HEALTHZ_HEAD")
    logger.info("❤️ HEAD /healthz endpoint hit.", extra={"request_id": req_id})
    return Response(status_code=200)

@app.api_route("/health", methods=["GET", "HEAD"], status_code=200, summary="Detailed Health Check")
async def health_detailed(request_id: str = Depends(get_request_id)):
    logger.debug("💙 Detailed /health endpoint called.", extra={"request_id": request_id})
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "app_instance_id": APP_INSTANCE_ID, "request_id": request_id}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_route(
    data: AnalyzeRequest,
    request_id: str = Depends(get_request_id)
):
    start_time = time.perf_counter()
    logger.info(f"Received /analyze request. Grid shape: {len(data.new_card)}x{len(data.new_card[0]) if data.new_card and data.new_card[0] else 'N/A'}",
                extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
    try:
        grid_np = np.array(data.new_card, dtype=int)
    except Exception as e:
        logger.error(f"Error converting new_card to NumPy array: {e}", exc_info=True, extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format in new_card: {str(e)}")
    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions: {grid_np.shape}", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=400, detail="Grid must be a 2D array and non-empty.")
    try:
        config_to_use = data.analyzer_config_override if data.analyzer_config_override else ANALYZER_INSTANCE_CONFIG
        suggestions_from_analyzer = analyzer.analyze_grid(
            grid_np,
            request_id=request_id,
            analyzer_config_override=config_to_use
        )
        response_suggestions: List[Suggestion] = [
            Suggestion(**sug) for sug in suggestions_from_analyzer
        ]
        duration_ms = (time.perf_counter() - start_time) * 1000
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
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"An unexpected error occurred during analysis. Duration: {duration_ms:.2f}ms: {e}", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")

@app.post("/score", response_model=ScoreModuleResponse)
async def score_module_route(
    data: ScoreModuleRequest,
    request_id: str = Depends(get_request_id),
):
    start_time = time.perf_counter()
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
        module_pydantic_config_class = type(brain.DEFAULT_MODULE_CONFIGS.get(data.module_name, brain.BaseModuleConfig()))
        final_module_config: brain.BaseModuleConfig
        if data.module_config_override:
            try:
                final_module_config = module_pydantic_config_class(**data.module_config_override)
                logger.info(f"Using overridden config for module {data.module_name}: {final_module_config.model_dump_json()}",
                            extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
            except Exception as pydantic_error:
                logger.error(f"Invalid module_config_override for {data.module_name}: {pydantic_error}", exc_info=True, extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
                raise HTTPException(status_code=400, detail=f"Invalid config override for module {data.module_name}: {pydantic_error}")
        else:
            final_module_config = brain.DEFAULT_MODULE_CONFIGS.get(data.module_name, brain.BaseModuleConfig())
            logger.info(f"Using default config for module {data.module_name}.", extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})

        score_matrix = brain.get_module_score(
            data.module_name,
            grid_np,
            config_override=final_module_config,
            request_id=request_id
        )
        preview: List[List[float]] | None = None
        if score_matrix.size > 0:
            preview_rows = min(score_matrix.shape[0], 5)
            preview_cols = min(score_matrix.shape[1], 5)
            if preview_rows > 0 and preview_cols > 0:
                 preview = score_matrix[:preview_rows, :preview_cols].tolist()
        duration_ms = (time.perf_counter() - start_time) * 1000
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
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"An unexpected error occurred while scoring with module {data.module_name}. Duration: {duration_ms:.2f}ms: {e}",
                         extra={"request_id": request_id, "app_instance_id": APP_INSTANCE_ID})
        raise HTTPException(status_code=500, detail=f"Internal server error during scoring: {str(e)}")

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    from app_settings import settings  # 請確認你的設定模組名稱
    import logging

    # 取得 uvicorn 的 log config
    log_config_uvicorn = uvicorn.config.LOGGING_CONFIG

    # 修改 formatter 格式，移除容易出錯的 request_id 等欄位
    log_config_uvicorn["formatters"]["default"]["fmt"] = (
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    )
    log_config_uvicorn["formatters"]["access"]["fmt"] = (
        '%(asctime)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
    )

    # 安全記錄啟動訊息（不會因 log formatter 崩潰）
    logger = logging.getLogger("uvicorn.error")
    logger.info(
        f"🚀 Starting Uvicorn server directly from main.py on port {settings.server_port} for {settings.app_name}..."
    )

    # 執行 Uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.server_port,
        log_config=log_config_uvicorn,
        reload=False,
    )
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.server_port, # Use port from settings
        log_level=settings.log_level.lower(),
        reload=True, # Good for development, set to False or remove for production
        log_config=log_config_uvicorn
    )