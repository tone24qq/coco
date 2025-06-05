"""
main14_optimized.py - 優化版 FastAPI 主程式，整合4模組分析器
"""
import time
import logging
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from analyzer11_optimized import analyze_with_prior

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

startup_time = datetime.now()
request_count = 0
total_process_time = 0.0

class AnalyzeRequest(BaseModel):
    """分析請求模型"""
    grid: List[List[int]] = Field(..., description="2D網格，-1表示空格")
    target: int = Field(..., description="目標數字")
    top_k: int = Field(3, ge=1, le=10, description="返回前K個結果")
    
    @validator('grid')
    def validate_grid(cls, v):
        if not v or not v[0]:
            raise ValueError("網格不能為空")
        rows = len(v)
        cols = len(v[0])
        if not all(len(row) == cols