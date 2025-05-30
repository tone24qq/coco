import asyncio
import time
import uuid
import logging
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path, Request, status, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKey, APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings
from starlette_prometheus import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter, Histogram, Gauge
from typing import Callable, Awaitable

# 舊寫法 ❌ from typing import Union, Optional
# 新寫法 ✅ 改為 PEP 604 標準
# 已全部使用 `|` 替代 Union/Optional

# 假設 brain 模組已實作好，並放置於相同目錄
import brain


class Settings(BaseSettings):
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    APP_TITLE: str = "橘子專案-進階自動補格評分API"
    APP_DESCRIPTION: str = "提供強化盤面評分模組的API接口，包含批次處理、背景任務與安全性增強。"
    APP_VERSION: str = "2.0.0"
    API_KEY: str = "changeme"
    API_KEY_NAME: str = "X-API-KEY"
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    TASK_CALLBACK_URL_ENABLED: bool = False
    TASK_CALLBACK_URL: HttpUrl | None = None

    class Config:
        env_file = ".env"


settings = Settings()

# logging 設定：格式中不得出現 KeyError
class RequestIDLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s - RequestID: %(request_id)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")
logger.addFilter(RequestIDLogFilter())

# Prometheus metrics
REQUEST_COUNT = Counter("api_request_count", "Total API requests", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["method", "endpoint"])
ACTIVE_BACKGROUND_TASKS = Gauge("api_active_background_tasks", "Active background tasks")
MODULE_USAGE_COUNT = Counter("api_module_usage_count", "Module usage count", ["module_name"])

api_key_query = APIKeyQuery(name=settings.API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)


async def get_api_key(
    key_query: str | None = Security(api_key_query),
    key_header: str | None = Security(api_key_header),
) -> str:
    if key_query == settings.API_KEY or key_header == settings.API_KEY:
        return settings.API_KEY
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")


class GridDataBase(BaseModel):
    grid_data: list[list[int | float]] = Field(..., example=[[1, -1], [2, 3]])

    @field_validator("grid_data")
    @classmethod
    def validate_grid(cls, v: list[list[int | float]]) -> list[list[int | float]]:
        if not v or not all(isinstance(row, list) for row in v) or not v[0]:
            raise ValueError("grid_data must be a non-empty list of lists")
        num_cols = len(v[0])
        if any(len(row) != num_cols for row in v):
            raise ValueError("All rows must have the same number of columns")
        return v


class GridInput(GridDataBase):
    client_request_id: str | None = None


class BatchGridItem(GridDataBase):
    item_id: str
    module_name: str


class BatchGridInput(BaseModel):
    grids: list[BatchGridItem]
    client_request_id: str | None = None


class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: str = "accepted"
    message: str
    client_request_id: str | None = None


class ModuleInfo(BaseModel):
    name: str
    description: str | None = "No description available."
    version: str | None = "N/A"


request_counts: dict[str, list[float]] = {}

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

@app.middleware("http")
async def base_middleware(request: Request, call_next: Callable[[Request], Awaitable]) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    request_counts[client_ip] = [t for t in request_counts.get(client_ip, []) if t > now - settings.RATE_LIMIT_WINDOW_SECONDS]
    if len(request_counts[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=429).inc()
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})

    request_counts.setdefault(client_ip, []).append(now)
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
async def root(request: Request):
    return {"message": f"Welcome to {settings.APP_TITLE}", "version": settings.APP_VERSION}


@app.get("/modules", response_model=list[ModuleInfo])
async def list_modules(api_key: APIKey = Depends(get_api_key)) -> list[ModuleInfo]:
    modules = []
    for name in brain.REGISTERED_MODULES_BRAIN:
        details = getattr(brain, "get_module_details", lambda name: {})(name)
        modules.append(ModuleInfo(name=name, description=details.get("description"), version=details.get("version")))
    return modules


@app.post("/score/{module_name}", response_model=TaskAcceptedResponse, status_code=202)
async def score_grid(
    request: Request,
    payload: GridInput,
    module_name: str = Path(...),
    background_tasks: BackgroundTasks = Depends(),
    api_key: APIKey = Depends(get_api_key),
) -> TaskAcceptedResponse:
    task_id = str(uuid.uuid4())
    if module_name not in brain.REGISTERED_MODULES_BRAIN:
        raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found")
    background_tasks.add_task(run_scoring_task, task_id, module_name, payload.grid_data, request.state.request_id, payload.client_request_id)
    MODULE_USAGE_COUNT.labels(module_name=module_name).inc()
    return TaskAcceptedResponse(task_id=task_id, message="Task accepted", client_request_id=payload.client_request_id)


async def run_scoring_task(task_id: str, module_name: str, grid_data: list[list[int | float]], request_id: str, client_id: str | None = None) -> None:
    ACTIVE_BACKGROUND_TASKS.inc()
    try:
        np_grid = np.array(grid_data, dtype=np.float32)
        score = await asyncio.to_thread(brain.get_module_score, module_name, np_grid, request_id=task_id)
        logger.info(f"Task {task_id} complete", extra={"request_id": request_id})
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", extra={"request_id": request_id}, exc_info=True)
    finally:
        ACTIVE_BACKGROUND_TASKS.dec()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
