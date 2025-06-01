# main.py
# 本文件自動生成，依據新大腦.pdf、給你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 系統入口與 FastAPI 路由層。

import os
import logging
import uuid
from datetime import datetime, timezone
import time # For performance logging
import asyncio # 新增：為了 keep_alive_task

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, Response # Response 新增
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Tuple, cast
from typing import Callable
from dotenv import load_dotenv

load_dotenv()

import analyzer
import brain

# --- Configuration via Pydantic BaseSettings ---
class AppSettings(BaseSettings):
    app_name: str = Field(default="AI Scoring Service", validation_alias="APP_NAME")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

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
logger = RequestIdLoggerAdapter(_base_logger, {'request_id': 'APP_DEFAULT_ID'})


# --- Keep Alive Task (inspired by 測試.txt) ---
async def keep_alive_task_main_app():
    """每60秒打印一次日誌以保持服務活躍，避免被Render等平台視為idle而關閉。"""
    while True:
        logger.info("💡 Main App Still alive... (avoiding idle shutdown)", extra={"request_id": "keep_alive"})
        await asyncio.sleep(60)


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Module Scoring Service based on a 3-tier architecture (main -> analyzer -> brain)."
)

# --- FastAPI Event Handlers ---
ANALYZER_INSTANCE_CONFIG: analyzer.AnalyzerConfig

@app.on_event("startup")
async def startup_event():
    global ANALYZER_INSTANCE_CONFIG
    logger.info(f"Application '{settings.app_name}' starting up...", extra={"request_id": "startup"})
    logger.info(f"Log level set to: {settings.log_level}", extra={"request_id": "startup"})
    
    try:
        ANALYZER_INSTANCE_CONFIG = analyzer.DEFAULT_ANALYZER_CONFIG
        analyzer.initialize_analyzer(config_override=ANALYZER_INSTANCE_CONFIG)
        logger.info("Analyzer initialized successfully.", extra={"request_id": "startup"})
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize analyzer during startup: {e}", exc_info=True, extra={"request_id": "startup_CRITICAL_ERROR"})
        # Depending on severity, you might want to raise an error here to prevent app from starting in a broken state,
        # or allow it to start with limited functionality if possible.
        # For now, it logs the error and continues, but Render might see it as unhealthy if critical parts fail.

    # Start the keep-alive task
    asyncio.create_task(keep_alive_task_main_app()) # [cite: 1]
    logger.info("Keep-alive task started.", extra={"request_id": "startup"})
    logger.info("Application startup complete.", extra={"request_id": "startup"})

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...", extra={"request_id": "shutdown"})
    # Add any other cleanup logic here
    logger.info("Application shutdown complete.", extra={"request_id": "shutdown"})


# --- Request ID Middleware and Dependency ---
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

async def get_request_id(request: Request) -> str:
    return cast(str, getattr(request.state, "request_id", "unknown_request_id"))

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (assuming these are defined as before) ---
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

# 修正：明確支援 GET 和 HEAD 方法，參考 測試.txt [cite: 1]
@app.api_route("/", methods=["GET", "HEAD"])
async def root(request: Request): # Added request for logging context
    req_id = getattr(request.state, "request_id", "NO_ID_IN_ROOT")
    logger.info("Root / endpoint hit.", extra={"request_id": req_id})
    return {"message": "Service is alive."}

# 修正：確保 /healthz GET 正常，並明確添加 HEAD 支持，參考 測試.txt [cite: 2]
@app.get("/healthz")
async def health_check_z(request: Request): # Renamed from health_check
    req_id = getattr(request.state, "request_id", "NO_ID_IN_HEALTHZ")
    logger.info("GET /healthz endpoint hit.", extra={"request_id": req_id})
    return {"status": "ok"}

@app.api_route("/healthz", methods=["HEAD"]) # Explicit HEAD handler [cite: 2]
async def health_check_z_head(request: Request):
    req_id = getattr(request.state, "request_id", "NO_ID_IN_HEALTHZ_HEAD")
    logger.info("HEAD /healthz endpoint hit.", extra={"request_id": req_id})
    return Response(status_code=200) # HEAD should return 200 OK with no body

# 保留 /health 端點，並明確支援 GET 和 HEAD
@app.api_route("/health", methods=["GET", "HEAD"], status_code=200, summary="Detailed Health Check")
async def health_detailed_check(request_id: str = Depends(get_request_id)): # Renamed from health_check to avoid conflict
    logger.debug("Detailed /health endpoint called.", extra={"request_id": request_id})
    # For HEAD, FastAPI/Starlette should automatically strip the body if we return content
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_route(
    data: AnalyzeRequest,
    request_id: str = Depends(get_request_id)
):
    start_time = time.perf_counter()
    logger.info(f"Received /analyze request. Grid shape: {len(data.new_card)}x{len(data.new_card[0]) if data.new_card and data.new_card[0] else 'N/A'}",
                extra={"request_id": request_id})
    try:
        grid_np = np.array(data.new_card, dtype=int)
    except Exception as e:
        logger.error(f"Error converting new_card to NumPy array: {e}", exc_info=True, extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format in new_card: {str(e)}")
    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions: {grid_np.shape}", extra={"request_id": request_id})
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
                    extra={"request_id": request_id})
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
        logger.exception(f"An unexpected error occurred during analysis. Duration: {duration_ms:.2f}ms: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")

@app.post("/score", response_model=ScoreModuleResponse)
async def score_module_route(
    data: ScoreModuleRequest,
    request_id: str = Depends(get_request_id),
):
    start_time = time.perf_counter()
    logger.info(f"Received /score request for module: {data.module_name}", extra={"request_id": request_id})
    if data.module_name not in brain.REGISTERED_MODULES_BRAIN:
        logger.warning(f"Module '{data.module_name}' not found.", extra={"request_id": request_id})
        raise HTTPException(status_code=404, detail=f"Module '{data.module_name}' not found.")
    try:
        grid_np = np.array(data.grid_data, dtype=int)
    except Exception as e:
        logger.error(f"Error converting grid_data to NumPy array: {e}", exc_info=True, extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format: {str(e)}")
    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions for scoring: {grid_np.shape}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail="Grid for scoring must be a 2D array and non-empty.")
    try:
        module_pydantic_config_class = type(brain.DEFAULT_MODULE_CONFIGS.get(data.module_name, brain.BaseModuleConfig()))
        final_module_config: brain.BaseModuleConfig
        if data.module_config_override:
            try:
                final_module_config = module_pydantic_config_class(**data.module_config_override)
                logger.info(f"Using overridden config for module {data.module_name}: {final_module_config.model_dump_json()}",
                            extra={"request_id": request_id})
            except Exception as pydantic_error:
                logger.error(f"Invalid module_config_override for {data.module_name}: {pydantic_error}", exc_info=True, extra={"request_id": request_id})
                raise HTTPException(status_code=400, detail=f"Invalid config override for module {data.module_name}: {pydantic_error}")
        else:
            final_module_config = brain.DEFAULT_MODULE_CONFIGS.get(data.module_name, brain.BaseModuleConfig())
            logger.info(f"Using default config for module {data.module_name}.", extra={"request_id": request_id})

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
        logger.info(f"Successfully scored grid with module '{data.module_name}'. Duration: {duration_ms:.2f}ms", extra={"request_id": request_id})
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
                         extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Internal server error during scoring: {str(e)}")

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    log_config_uvicorn = uvicorn.config.LOGGING_CONFIG
    log_config_uvicorn["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config_uvicorn["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'

    logger.info(f"Starting Uvicorn server directly from main.py on port 8000 for {settings.app_name}...",
                extra={"request_id": "main_direct_run"})
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        reload=True,
        log_config=log_config_uvicorn
    )