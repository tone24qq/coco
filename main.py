# main.py
# 本文件自動生成，依據新大腦.pdf、給你2025资料在深度建议一次.pdf、极限强化.pdf 維度實現
# 系統入口與 FastAPI 路由層。

import os
import logging
import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Tuple

# 來源：analyzer.py, brain.py (本项目)
import analyzer
import brain # Though main doesn't call brain directly, analyzer does.

# --- Configuration via Pydantic BaseSettings ---
# 來源：main.py (用户需求 Point 4.a)
# 來源：给你2025资料在深度建议一次.pdf - Pydantic V2 BaseSettings (Page 3 section 3.1.2)
class AppSettings(BaseSettings):
    app_name: str = "AI Scoring Service"
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    # Add other settings like DB URLs, cache configs if needed
    # example_api_key: str = Field(validation_alias="EXAMPLE_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields from .env

settings = AppSettings()

# --- Logging Setup ---
# 來源：main.py (用户需求 Point 4.c)
# 來源：请给出26个模块极限强化（针对手机版）的深度意见分析方向越详细越好.pdf - 日誌與監控整合
# 來源：给你2025资料在深度建议一次.pdf - 日誌與監控整合 (Page 1)
# Basic logging config, can be enhanced with structlog or other libraries
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
)
logger = logging.getLogger(settings.app_name)

# Adapter to inject request_id into log records
class RequestIdAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> Tuple[str, Any]:
        request_id = kwargs.pop("request_id", self.extra.get("request_id"))
        return f"[{request_id}] {msg}", kwargs

# --- FastAPI App Initialization ---
# 來源：main.py (用户需求 Point 1, 4)
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI Module Scoring Service based on a 3-tier architecture (main -> analyzer -> brain)."
)

# CORS Middleware (optional, good for development)
# 來源：2024-2025新知識.txt - FastAPI CORS (Page 8 section 3.1.3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request ID Middleware and Dependency ---
# 來源：main.py (用户需求 Point 4.c, 4.e)
# 來源：指令.txt - 引用logging/request_id (Point 3)
class RequestIDMiddleware:
    async def __call__(self, request: Request, call_next: Any):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Make request_id available to the logger adapter and dependencies
        request.state.request_id = request_id
        
        # For logger adapter to pick up request_id from task locals if not passed explicitly
        # This part would require a more complex contextvar setup for logging,
        # for simplicity, we'll rely on passing it via `extra` or using the adapter manually.
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.middleware("http")(RequestIDMiddleware())

async def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown_request")


# --- Pydantic Models for API ---
# 來源：main.py (用户需求 Point 4.a, 4.e)
# 來源：给你2025资料在深度建议一次.pdf - Pydantic V2 (Page 1) & PEP 604 (Page 1)
class GridCell(BaseModel):
    r: int
    c: int
    value: int

class AnalyzeRequest(BaseModel):
    # new_card represents the current state of the board
    new_card: List[List[int]] = Field(..., description="二維陣列代表當前盤面，-1表示空格")
    # proposed_values is mentioned in prompt, but analyzer primarily scores empty cells.
    # It might be used if we want to evaluate specific placements, not covered by current analyzer logic.
    # For now, it's optional or can be used by a different endpoint/logic.
    proposed_values: Dict[str, List[int]] | None = Field(default=None, description="提議的值 (目前未使用於主要分析邏輯)")

    @validator('new_card')
    def check_grid_not_empty_and_rectangular(cls, v: List[List[int]]):
        if not v:
            raise ValueError("new_card (grid) cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("new_card must be a list of lists")
        if not v[0]: # Assuming at least one row from previous check
             raise ValueError("new_card rows cannot be empty")
        
        row_len = len(v[0])
        if not all(len(row) == row_len for row in v):
            raise ValueError("All rows in new_card must have the same length")
        return v

class Suggestion(BaseModel):
    coords: Tuple[int, int]
    confidence_score: float
    contributing_modules: Dict[str, float] | None = None


class AnalyzeResponse(BaseModel):
    request_id: str
    timestamp: str
    suggestions: List[Suggestion]
    grid_shape_analyzed: Tuple[int, int]
    message: str | None = None


class ScoreModuleRequest(BaseModel):
    module_name: str
    grid_data: List[List[int]] = Field(..., description="二維陣列代表盤面")

    @validator('grid_data')
    def check_score_grid(cls, v: List[List[int]]):
        # Similar validation as AnalyzeRequest.new_card
        if not v or not v[0] or not all(len(row) == len(v[0]) for row in v):
            raise ValueError("grid_data must be a non-empty rectangular list of lists")
        return v

class ScoreModuleResponse(BaseModel):
    request_id: str
    module_name: str
    # Score grid can be large, consider how to represent it if needed, or just a summary
    # For now, let's assume we might not return the full grid, but a message.
    # Or if small, can be List[List[float]]
    scores_preview: List[List[float]] | None = None # Preview of scores
    message: str


# --- FastAPI Event Handlers ---
# 來源：main.py (用户需求 Point 4.b)
@app.on_event("startup")
async def startup_event():
    logger.info(f"Application '{settings.app_name}' starting up...")
    logger.info(f"Log level set to: {settings.log_level}")
    analyzer.initialize_analyzer() # Initialize analyzer, load configs/models if any
    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    # Cleanup resources if any
    logger.info("Application shutdown complete.")


# --- API Endpoints ---
# 來源：main.py (用户需求 Point 4)
# 來源：三層結構崗位說明.txt - main.py 職責
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_route(
    data: AnalyzeRequest,
    request: Request, # To access request.state.request_id if middleware sets it
    # Alternatively, use a dependency: request_id_dep: str = Depends(get_request_id)
):
    """
    Analyzes the provided grid (new_card) and returns Top-N suggested empty cells to fill.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())) # Ensure request_id
    logger.info(f"Received /analyze request. Grid dimensions: {len(data.new_card)}x{len(data.new_card[0]) if data.new_card else 0}", 
                extra={"request_id": request_id})

    try:
        grid_np = np.array(data.new_card, dtype=int)
    except ValueError as e:
        logger.error(f"Error converting new_card to NumPy array: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format: {e}")

    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions: {grid_np.shape}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail="Grid must be a 2D array and non-empty.")

    try:
        # 來源：main.py (用户需求 Point 4.a) - 呼叫 analyzer.py 的分析函數
        # Pass the default analyzer config, or load/create one here if dynamic
        suggestions_from_analyzer = analyzer.analyze_grid(
            grid_np, 
            request_id=request_id,
            analyzer_config=analyzer.DEFAULT_ANALYZER_CONFIG # Or a dynamically loaded one
        )
        
        # Format suggestions for the response model
        response_suggestions: List[Suggestion] = []
        for sug in suggestions_from_analyzer:
            response_suggestions.append(
                Suggestion(
                    coords=sug["coords"],
                    confidence_score=sug["confidence_score"],
                    contributing_modules=sug.get("contributing_modules") # Optional
                )
            )

        logger.info(f"Successfully analyzed grid. Found {len(response_suggestions)} suggestions.", 
                    extra={"request_id": request_id})
        return AnalyzeResponse(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            suggestions=response_suggestions,
            grid_shape_analyzed=grid_np.shape,
            message="Analysis successful."
        )
    # 來源：main.py (用户需求 Point 4.d) - 異常捕獲
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during analysis: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Internal server error during analysis: {str(e)}")


# 來源：main.py (用户需求 Point 4 - /score (可选))
@app.post("/score", response_model=ScoreModuleResponse)
async def score_module_route(
    data: ScoreModuleRequest,
    request: Request,
):
    """
    Scores a grid using a single specified brain module.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"Received /score request for module: {data.module_name}", extra={"request_id": request_id})

    if data.module_name not in brain.REGISTERED_MODULES_BRAIN:
        logger.warning(f"Module '{data.module_name}' not found.", extra={"request_id": request_id})
        raise HTTPException(status_code=404, detail=f"Module '{data.module_name}' not found.")

    try:
        grid_np = np.array(data.grid_data, dtype=int)
    except ValueError as e:
        logger.error(f"Error converting grid_data to NumPy array: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid grid_data format: {e}")
    
    if grid_np.ndim != 2 or grid_np.size == 0:
        logger.error(f"Invalid grid dimensions for scoring: {grid_np.shape}", extra={"request_id": request_id})
        raise HTTPException(status_code=400, detail="Grid for scoring must be a 2D array and non-empty.")

    try:
        # Retrieve potential Pydantic config for the module
        # For simplicity, using default config from AnalyzerConfig if available, or None
        module_pydantic_config = analyzer.get_module_specific_config(data.module_name, analyzer.DEFAULT_ANALYZER_CONFIG)

        score_matrix = brain.get_module_score(
            data.module_name, 
            grid_np, 
            config=module_pydantic_config, 
            request_id=request_id
        )
        
        # For response, maybe return a small preview if matrix is large
        preview: List[List[float]] | None = None
        if score_matrix.size > 0:
            preview_rows = min(score_matrix.shape[0], 5)
            preview_cols = min(score_matrix.shape[1], 5)
            preview = score_matrix[:preview_rows, :preview_cols].tolist()


        logger.info(f"Successfully scored grid with module '{data.module_name}'.", extra={"request_id": request_id})
        return ScoreModuleResponse(
            request_id=request_id,
            module_name=data.module_name,
            scores_preview=preview,
            message=f"Grid scored successfully with {data.module_name}."
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred while scoring with module {data.module_name}: {e}", 
                         extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"Internal server error during scoring: {str(e)}")


# Health check endpoint
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# --- Main execution for Uvicorn ---
# 來源：main.py (用户需求 Point 6) - python main.py 能成功启动
if __name__ == "__main__":
    import uvicorn
    # This allows running with `python main.py`
    # For production, prefer `uvicorn main:app --host 0.0.0.0 --port 8000`
    logger.info("Starting Uvicorn server directly from main.py...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.log_level.lower())