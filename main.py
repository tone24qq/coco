# main.py
# 本文件自動生成，依據新大腦.pdf、給你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 系統入口與 FastAPI 路由層。

import os
import logging
import uuid
from datetime import datetime, timezone
import time # For performance logging

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings # 來源：main.py (用户需求 Point 4.a) & 给你2025资料在深度建议一次.pdf (Page 3 section 3.1.2)
from typing import List, Dict, Any, Tuple, cast
from typing import Callable

# 來源：analyzer.py, brain.py (本项目)
import analyzer
import brain # Though main doesn't call brain directly, analyzer uses its DEFAULT_MODULE_CONFIGS.

# --- Configuration via Pydantic BaseSettings ---
# 來源：main.py (用户需求 Point 4.a)
class AppSettings(BaseSettings):
    app_name: str = Field(default="AI Scoring Service", validation_alias="APP_NAME")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    # 來源：analyzer.py (用户需求 Point 3.e) - main.py can load analyzer config
    # This allows main.py to potentially load AnalyzerConfig from a different source if needed
    # For now, analyzer.py defines its own DEFAULT_ANALYZER_CONFIG.
    # If main.py were to control it, it would instantiate analyzer.AnalyzerConfig here,
    # possibly loading its values from .env as well.
    # Example: analyzer_config_json: str | None = Field(default=None, validation_alias="ANALYZER_CONFIG_JSON")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = AppSettings()

# --- Logging Setup ---
# 來源：main.py (用户需求 Point 4.c)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 日誌與監控整合
# 來源：给你2025资料在深度建议一次.pdf - 日誌與監控整合 (Page 1)
# Basic logging config, can be enhanced with structlog or other libraries
# The format includes a placeholder for request_id
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"  # 🔧 先拿掉 request_id
)
logger = logging.getLogger(settings.app_name)

# Adapter to inject request_id into log records if not explicitly passed
# This is a simple way; for more robust context, contextvars might be used.
class RequestIdLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> Tuple[str, Any]:
        # Ensure 'extra' exists and has 'request_id'
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # If request_id is already in extra, use it, otherwise use the adapter's default.
        # The default set in the adapter might come from request.state.
        current_request_id = kwargs['extra'].get('request_id', self.extra.get('request_id', "NO_REQUEST_ID"))
        kwargs['extra']['request_id'] = current_request_id
        
        # The format string in basicConfig now directly uses %(request_id)s,
        # so we just need to ensure 'request_id' is in the record's dictionary.
        # The adapter's role here is more about ensuring it's present.
        return msg, kwargs


# --- FastAPI App Initialization ---
# 來源：main.py (用户需求 Point 1, 4)
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Module Scoring Service based on a 3-tier architecture (main -> analyzer -> brain)."
)
@app.get("/")
async def root():
    return {"message": "Service is alive.
@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
# CORS Middleware
# 來源：2024-2025新知識.txt - FastAPI CORS (Page 8 section 3.1.3) [Internal Alias]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request ID Middleware and Dependency ---
# 來源：main.py (用户需求 Point 4.c, 4.e) & 指令.txt - 引用logging/request_id (Point 3)
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable):
    # Try to get X-Request-ID header, or generate a new one
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id # Store it in request state

    # For logging: pass request_id explicitly via `extra` when logging
    # Or configure logger/handlers to pick it up from task-local storage if using contextvars

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id # Add it to the response
    return response

async def get_request_id(request: Request) -> str:
    return cast(str, getattr(request.state, "request_id", "unknown_request_id"))


# --- Pydantic Models for API ---
# 來源：main.py (用户需求 Point 4.a, 4.e)
# 來源：给你2025资料在深度建议一次.pdf - Pydantic V2 (Page 1) & PEP 604 (Page 1)
class AnalyzeRequest(BaseModel):
    new_card: List[List[int]] = Field(..., description="二維陣列代表當前盤面，-1表示空格")
    # proposed_values is kept as per prompt, though current analyzer doesn't use it.
    # It could be used for "what-if" scenarios if analyzer logic is extended.
    proposed_values: Dict[str, List[int]] | None = Field(
        default=None, 
        description="提議的值 (目前主要分析邏輯未使用，可為特定擴展保留)"
    )
    # Allow passing an override for AnalyzerConfig if needed for a specific request
    analyzer_config_override: analyzer.AnalyzerConfig | None = Field(default=None, description="可選：覆蓋預設的分析器設定")


    @validator('new_card')
    def check_grid_not_empty_and_rectangular(cls, v: List[List[int]]) -> List[List[int]]:
        if not v:
            raise ValueError("new_card (grid) cannot be empty")
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("new_card must be a list of lists of integers")
        
        if not v[0]: # First row cannot be empty if grid is not empty
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
    duration_ms: float # For performance monitoring


class ScoreModuleRequest(BaseModel):
    module_name: str = Field(...)
    grid_data: List[List[int]] = Field(..., description="二維陣列代表盤面")
    # Allow passing a specific Pydantic config for the module being scored
    module_config_override: Dict[str, Any] | None = Field(default=None, description="可選：覆蓋該模組的預設Pydantic設定（JSON對象）")


    @validator('grid_data')
    def check_score_grid(cls, v: List[List[int]]) -> List[List[int]]:
        # Similar validation as AnalyzeRequest.new_card
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


# --- FastAPI Event Handlers ---
# 來源：main.py (用户需求 Point 4.b)
ANALYZER_INSTANCE_CONFIG: analyzer.AnalyzerConfig # Global for this main instance

@app.on_event("startup")
async def startup_event():
    global ANALYZER_INSTANCE_CONFIG
    # Here, one could load AnalyzerConfig from a file or other sources.
    # For now, we use the default defined in analyzer.py.
    # If AppSettings had analyzer_config_json, we could load it:
    # if settings.analyzer_config_json:
    #     try:
    #         ANALYZER_INSTANCE_CONFIG = analyzer.AnalyzerConfig.model_validate_json(settings.analyzer_config_json)
    #         logger.info("Loaded AnalyzerConfig from AppSettings (JSON).")
    #     except Exception as e:
    #         logger.error(f"Failed to load AnalyzerConfig from JSON: {e}. Using default.")
    #         ANALYZER_INSTANCE_CONFIG = analyzer.DEFAULT_ANALYZER_CONFIG
    # else:
    ANALYZER_INSTANCE_CONFIG = analyzer.DEFAULT_ANALYZER_CONFIG # Use default from analyzer.py
    
    analyzer.initialize_analyzer(config_override=ANALYZER_INSTANCE_CONFIG)
    
    logger.info(f"Application '{settings.app_name}' starting up...", extra={"request_id": "startup"})
    logger.info(f"Log level set to: {settings.log_level}", extra={"request_id": "startup"})
    logger.info("Application startup complete.", extra={"request_id": "startup"})

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...", extra={"request_id": "shutdown"})
    logger.info("Application shutdown complete.", extra={"request_id": "shutdown"})


# --- API Endpoints ---
# 來源：main.py (用户需求 Point 4)
# 來源：三層結構崗位說明.txt - main.py 職責
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_route(
    data: AnalyzeRequest,
    # 來源：main.py (用户需求 Point 4.e) - Depends for request_id
    request_id: str = Depends(get_request_id) 
):
    """
    Analyzes the provided grid (new_card) and returns Top-N suggested empty cells to fill.
    """
    start_time = time.perf_counter()
    logger.info(f"Received /analyze request. Grid shape: {len(data.new_card)}x{len(data.new_card[0]) if data.new_card and data.new_card[0] else 'N/A'}", 
                extra={"request_id": request_id})

    try:
        grid_np = np.array(data.new_card, dtype=int)
    except Exception as e: # More general exception for array conversion
        logger.error(f"Error converting new_card to NumPy array: {e}", exc_info=True, extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format in new_card: {str(e)}")

    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions: {grid_np.shape}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail="Grid must be a 2D array and non-empty.")

    try:
        # Use the AnalyzerConfig from the request if provided, else use the instance default
        config_to_use = data.analyzer_config_override if data.analyzer_config_override else ANALYZER_INSTANCE_CONFIG
        
        # 來源：main.py (用户需求 Point 4.a) - 呼叫 analyzer.py 的分析函數
        suggestions_from_analyzer = analyzer.analyze_grid(
            grid_np, 
            request_id=request_id,
            analyzer_config_override=config_to_use 
        )
        
        response_suggestions: List[Suggestion] = [
            Suggestion(**sug) for sug in suggestions_from_analyzer # Directly unpack if keys match
        ]
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Successfully analyzed grid. Found {len(response_suggestions)} suggestions. Duration: {duration_ms:.2f}ms", 
                    extra={"request_id": request_id})
        return AnalyzeResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            suggestions=response_suggestions,
            grid_shape_analyzed=cast(Tuple[int, int], grid_np.shape), # Cast for type checker
            message="Analysis successful.",
            duration_ms=duration_ms
        )
    # 來源：main.py (用户需求 Point 4.d) - 異常捕獲
    except HTTPException: # Re-raise HTTPExceptions from analyzer or validation
        raise
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"An unexpected error occurred during analysis. Duration: {duration_ms:.2f}ms: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")


# 來源：main.py (用户需求 Point 4 - /score (可选))
@app.post("/score", response_model=ScoreModuleResponse)
async def score_module_route(
    data: ScoreModuleRequest,
    request_id: str = Depends(get_request_id),
):
    """
    Scores a grid using a single specified brain module.
    Allows overriding the module's default Pydantic configuration.
    """
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
        # Get the default Pydantic config class for this module from brain.py
        module_pydantic_config_class = type(brain.DEFAULT_MODULE_CONFIGS.get(data.module_name, brain.BaseModuleConfig()))
        
        final_module_config: brain.BaseModuleConfig
        if data.module_config_override:
            try:
                # Create an instance from the override, validating against the module's specific Pydantic class
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


# Health check endpoint
@app.get("/health", status_code=200, summary="Health Check")
async def health_check(request_id: str = Depends(get_request_id)):
    logger.debug("Health check endpoint called.", extra={"request_id": request_id})
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id}


# --- Main execution for Uvicorn ---
# 來源：main.py (用户需求 Point 6) - python main.py 能成功启动
if __name__ == "__main__":
    import uvicorn
    # This allows running with `python main.py`
    # For production, prefer `uvicorn main:app --host 0.0.0.0 --port 8000 [other_options]`
    # The AppSettings log_level is already applied to the root logger.
    # Uvicorn's own log level can also be set.
    log_config_uvicorn = uvicorn.config.LOGGING_CONFIG
    log_config_uvicorn["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s" # Simpler for uvicorn itself
    log_config_uvicorn["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelname)s - %(client_addr)s - "%(request_line)s" %(status_code)s'


    logger.info(f"Starting Uvicorn server directly from main.py on port 8000 for {settings.app_name}...", 
                extra={"request_id": "main_direct_run"})
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level=settings.log_level.lower(),
        reload=True # Good for development, remove for production
        # log_config=log_config_uvicorn # Optional: custom uvicorn log format
    )