# main.py
# 系统入口与 FastAPI 路由层 - zero errors, full logging, request_id, background tasks, etc.

import os
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Callable, Tuple, Any, Dict, List

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

import analyzer
import brain1
import brain2
import brain3

from dotenv import load_dotenv

load_dotenv()


class AppSettings(BaseSettings):
    """
    Application settings for FastAPI.
    """
    APP_NAME: str = Field(default="ScratchcardAnalyzerAPI", env="APP_NAME")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    PORT: int = Field(default=8000, env="PORT")
    SELF_PING_URL: str = Field(
        default="http://localhost:8000/healthz", env="SELF_PING_URL"
    )
    ENABLE_CORS: bool = Field(default=True, env="ENABLE_CORS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = AppSettings()


class RequestIdLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that ensures every log record has a request_id in 'extra'.
    """
    def process(self, msg: str, kwargs: Dict) -> Tuple[str, Dict]:
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        current_request_id = kwargs["extra"].get(
            "request_id", self.extra.get("request_id", "NO_REQUEST_ID")
        )
        kwargs["extra"]["request_id"] = current_request_id
        return msg, kwargs


logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - [%(request_id)s] - %(message)s",
)
_base_logger = logging.getLogger(settings.APP_NAME)
logger = RequestIdLoggerAdapter(_base_logger, {"request_id": f"APP-{uuid.uuid4()}"})


async def keep_alive_log():
    """
    每60秒打印一次日志以保持服务活跃。
    """
    while True:
        logger.info("💓 Keep-alive heartbeat", extra={"request_id": "keep_alive"})
        await asyncio.sleep(60)


async def self_ping():
    """
    每60秒向自身 /healthz 端点发起 GET 请求，产生真实流量（适用于 Render 等平台）。
    """
    url = settings.SELF_PING_URL
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                logger.info(
                    f"🩺 Self-ping to {url} SUCCESS, status {resp.status_code}",
                    extra={"request_id": "self_ping"},
                )
            except httpx.RequestError as e:
                logger.error(
                    f"🩺 Self-ping RequestError: {e}",
                    extra={"request_id": "self_ping_error"},
                    exc_info=False,
                )
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"🩺 Self-ping HTTPStatusError {e.response.status_code}: {e}",
                    extra={"request_id": "self_ping_error"},
                    exc_info=False,
                )
            except Exception as e:
                logger.error(
                    f"🩺 Self-ping UNEXPECTED ERROR: {e}",
                    extra={"request_id": "self_ping_unexpected"},
                    exc_info=True,
                )
            await asyncio.sleep(60)


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI Scratch‐Card Analysis Service (Vectorized GM1–GM26)",
)


@app.on_event("startup")
async def on_startup():
    """
    启动时：记录启动日志、启动背景任务 keep_alive_log 与 self_ping。
    """
    logger.info("🚀 Application startup initiated", extra={"request_id": "startup"})
    asyncio.create_task(keep_alive_log())
    logger.info("✅ keep_alive_log task started", extra={"request_id": "startup"})
    asyncio.create_task(self_ping())
    logger.info("✅ self_ping task started", extra={"request_id": "startup"})
    logger.info("🏁 Application startup complete", extra={"request_id": "startup"})


@app.on_event("shutdown")
async def on_shutdown():
    """
    关闭时：记录关闭日志与 uptime。
    """
    logger.info("🛑 Application shutdown initiated", extra={"request_id": "shutdown"})
    logger.info("🏁 Application shutdown complete", extra={"request_id": "shutdown"})


@app.middleware("http")
async def inject_request_id(request: Request, call_next: Callable):
    """
    每次请求生成或读取 X-Request-ID，并将其放入 response header 及日志 extra 中。
    """
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


async def get_request_id(request: Request) -> str:
    return request.state.request_id  # type: ignore


if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )


@app.get("/healthz", tags=["health"])
async def healthz(request: Request):
    """
    简单健康检查，可用于 self-ping。
    """
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


class AnalyzeRequest(BaseModel):
    """
    请求数据模型：传入当前 scratch-card 网格与可选“提出的值”与“分析器配置”覆盖。
    """
    new_card: List[List[int]] = Field(..., description="2D 整数网格，每个 cell: 1~N 或 -1 表示空格")
    proposed_values: Dict[str, List[int]] | None = Field(
        default=None,
        description="可选：针对关键 modules 的候选值（GM1–GM26 可自定义）",
    )
    analyzer_config_override: Dict[str, Any] | None = Field(
        default=None, description="可选：覆盖默认分析配置"
    )

    @validator("new_card")
    def validate_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if not v:
            raise ValueError("new_card (grid) cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("new_card must be a list of lists")
        row_len = len(v[0])
        if row_len == 0:
            raise ValueError("new_card rows cannot be empty")
        for r_idx, row in enumerate(v):
            if len(row) != row_len:
                raise ValueError("All rows in new_card must be same length")
            for c_idx, cell in enumerate(row):
                if not isinstance(cell, int):
                    raise ValueError(
                        f"Cell ({r_idx},{c_idx}) must be int, found {type(cell)}"
                    )
        return v


class Candidate(BaseModel):
    """
    单个候选结果模型：cell 坐标，confidence 分数，module_scores 逐模块评分，reason 推理说明。
    """
    coords: Tuple[int, int]
    confidence_score: float
    module_scores: Dict[str, float]
    reason: str


class AnalyzeResponse(BaseModel):
    """
    返回结果：候选列表、完成 timestamp。
    """
    candidates: List[Candidate]
    timestamp: str


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze_endpoint(
    payload: AnalyzeRequest, request: Request
) -> AnalyzeResponse:
    """
    接收 POST /analyze，调用 analyzer.compute_combined_scores。
    """
    req_id = request.state.request_id  # type: ignore
    adapter = RequestIdLoggerAdapter(_base_logger, {"request_id": req_id})
    adapter.info("🔍 Received /analyze request", extra={"request_id": req_id})

    try:
        grid_np = np.array(payload.new_card, dtype=int)
    except Exception as e:
        adapter.error(f"Grid to NumPy conversion error: {e}", extra={"request_id": req_id})
        raise HTTPException(status_code=400, detail="Invalid new_card format")

    try:
        combined_scores = analyzer.compute_combined_scores(
            grid_np, payload.proposed_values or {}, req_id
        )
    except Exception as e:
        adapter.error(f"Analyzer internal error: {e}", extra={"request_id": req_id}, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error")

    candidates_list: List[Candidate] = []
    for (r, c), module_dict in combined_scores.items():
        confidence = float(sum(module_dict.values()) / len(module_dict)) if module_dict else 0.0
        reason = " | ".join(f"{m}={v:.3f}" for m, v in module_dict.items())
        candidates_list.append(
            Candidate(
                coords=(r, c),
                confidence_score=confidence,
                module_scores=module_dict,
                reason=reason,
            )
        )

    candidates_list.sort(key=lambda x: x.confidence_score, reverse=True)
    top3 = candidates_list[:3]

    adapter.info(f"🔍 /analyze returning top {len(top3)} candidates", extra={"request_id": req_id})
    return AnalyzeResponse(
        candidates=top3, timestamp=datetime.now(timezone.utc).isoformat()
    )