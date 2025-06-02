# main.py

import os
import logging
import asyncio
from typing import Optional, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from pydantic import Field
from pydantic_settings import BaseSettings

from analyzer import compute_combined_scores
from brain1 import (
    EXT_GM1_Proximity_Vec,
    EXT_GM2_Heterogeneity_Vec,
    EXT_GM3_PotentialField_Vec,
    BaseModuleConfig as GM1_3_Config,
)
from brain2 import (
    EXT_GM4_Spatial_Auto_Corr_Vec,
    EXT_GM5_Line_Completion_Vec,
    EXT_GM6_Symmetry_Potential_Vec,
    EXT_GM7_Numeric_Gaps_Vec,
    EXT_GM8_Edge_Affinity_Vec,
    EXT_GM9_Center_Control_Vec,
    EXT_GM10_BlockingValue_Vec,
    EXT_GM11_PairCorrelation_Vec,
    EXT_GM12_IslandAnalysis_Vec,
    BaseModuleConfig as GM4_12_Config,
)
from brain3 import (
    EXT_GM13_Sequence_Diversity_Vec,
    EXT_GM14_Risk_Assessment_Vec,
    EXT_GM15_Information_Gain_Vec,
    EXT_GM16_Harmonic_Centrality_Vec,
    EXT_GM17_Local_Entropy_Vec,
    EXT_GM18_RL_Value_Estimation_Vec,
    EXT_GM19_SkipPattern_Vec,
    EXT_GM20_SkipPattern_Confidence_Vec,
    EXT_GM21_ClusterBalance_Vec,
    EXT_GM22_CoOccurrence_Vec,
    EXT_GM23_MotifDetection_Vec,
    EXT_GM24_TemporalCoherence_Vec,
    EXT_GM25_StrategicDepth_Vec,
    EXT_GM26_ContextualFlexibility_Vec,
    BaseModuleConfig as GM13_26_Config,
)

logger = logging.getLogger("ScratchcardAnalyzerAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AppSettings(BaseSettings):
    APP_NAME: str = Field(default="ScratchcardAnalyzerAPI", env="APP_NAME")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    PORT: int = Field(default=8000, env="PORT")
    ENABLE_CORS: bool = Field(default=True, env="ENABLE_CORS")
    SELF_PING_URL: Optional[str] = Field(default=None, env="SELF_PING_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.SELF_PING_URL:
            self.SELF_PING_URL = f"http://localhost:{self.PORT}/healthz"


settings = AppSettings()
logger.setLevel(settings.LOG_LEVEL)


async def self_ping():
    """Periodically ping the application's health endpoint to keep it alive."""
    import httpx

    url = settings.SELF_PING_URL
    while True:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    logger.info(f"🩺 Self-ping to {url} SUCCESS, status {response.status_code}")
                else:
                    logger.error(f"🩺 Self-ping to {url} FAILED, status {response.status_code}")
        except Exception as e:
            logger.error(f"🩺 Self-ping RequestError: {e}")
        await asyncio.sleep(60)


async def keep_alive_log():
    """Log keep-alive heartbeat every minute."""
    while True:
        logger.info("💓 Keep-alive heartbeat")
        await asyncio.sleep(60)


app = FastAPI(title=settings.APP_NAME)

# Middlewares
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)

if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def on_startup():
    logger.info("[startup] 🚀 Application startup initiated")
    asyncio.create_task(keep_alive_log())
    asyncio.create_task(self_ping())
    logger.info("[startup] 🏁 Application startup complete")


@app.get("/healthz")
async def healthz():
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/analyze")
async def analyze(request: Request):
    try:
        payload = await request.json()
        grid = payload.get("grid")
        if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            raise ValueError("Invalid grid format")
        import numpy as np

        grid_array = np.array(grid, dtype=int)
    except Exception as e:
        logger.error(f"[analyze] 🛑 Invalid input: {e}")
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Construct configurations for GM1–GM26
    default_cfg_1_3 = GM1_3_Config(enabled=True, weight=1.0)
    default_cfg_4_12 = GM4_12_Config(enabled=True, weight=1.0)
    default_cfg_13_26 = GM13_26_Config(enabled=True, weight=1.0)

    module_configs = {
        "GM1": default_cfg_1_3,
        "GM2": default_cfg_1_3,
        "GM3": default_cfg_1_3,
        "GM4": default_cfg_4_12,
        "GM5": default_cfg_4_12,
        "GM6": default_cfg_4_12,
        "GM7": default_cfg_4_12,
        "GM8": default_cfg_4_12,
        "GM9": default_cfg_4_12,
        "GM10": default_cfg_4_12,
        "GM11": default_cfg_4_12,
        "GM12": default_cfg_4_12,
        "GM13": default_cfg_13_26,
        "GM14": default_cfg_13_26,
        "GM15": default_cfg_13_26,
        "GM16": default_cfg_13_26,
        "GM17": default_cfg_13_26,
        "GM18": default_cfg_13_26,
        "GM19": default_cfg_13_26,
        "GM20": default_cfg_13_26,
        "GM21": default_cfg_13_26,
        "GM22": default_cfg_13_26,
        "GM23": default_cfg_13_26,
        "GM24": default_cfg_13_26,
        "GM25": default_cfg_13_26,
        "GM26": default_cfg_13_26,
    }

    # Compute combined scores using the analyzer that calls PuzzleTensorOps.score_full_board
    try:
        results = compute_combined_scores(grid_array, module_configs, request_id=str(id(request)))
    except Exception as e:
        logger.error(f"[analyze] 🛑 Runtime error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return JSONResponse(status_code=200, content=results)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )