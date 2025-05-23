import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
from typing import List
import numpy as np
from celery.result import AsyncResult
from celery_worker import solve_task

app = FastAPI(title="Plug-in權重 + 張量流 + 自動數字範圍", version="3.1")

def get_card_max_value(grid):
    return max(v for row in grid for v in row if v != -1)

class ProposedValue(BaseModel):
    pos: List[int]
    value: int

class AnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

    @validator("new_card")
    def check_rectangular(cls, g):
        if not g or any(len(row) != len(g[0]) for row in g):
            raise ValueError("new_card 必须是矩形")
        return g

    @validator("proposed_values", each_item=True)
    def check_pv(cls, pv, values):
        grid = values.get("new_card")
        if grid:
            rows, cols = len(grid), len(grid[0])
            r, c = pv.pos
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"pos 越界：{pv.pos}")
            card_max = get_card_max_value(grid)
            if pv.value < 1 or pv.value > card_max:
                raise ValueError(f"value 超出卡片最大值範圍：1~{card_max}")
        return pv

def get_legal_values(grid: np.ndarray):
    card_max = get_card_max_value(grid)
    used = set(grid.flatten()[grid.flatten() != -1])
    return set(v for v in range(1, card_max+1) if v not in used)

@app.post("/analyze_async")
async def analyze_async(req: AnalyzeRequest):
    grid = np.array(req.new_card, dtype=int)
    legal_values = get_legal_values(grid)
    candidates = []
    for pv in req.proposed_values:
        r, c, v = pv.pos[0], pv.pos[1], pv.value
        if grid[r, c] != -1:
            continue
        if v in legal_values:
            candidates.append((r, c, v))
    if not candidates:
        raise HTTPException(400, "没有合法可选候选")
    task = solve_task.delay(grid.tolist(), candidates, list(legal_values))
    return {"task_id": task.id, "status": "processing"}

@app.get("/task_result/{task_id}")
def get_task_result(task_id: str):
    res = AsyncResult(task_id)
    if res.ready():
        return {"status": "completed", "result": res.result}
    else:
        return {"status": "processing"}