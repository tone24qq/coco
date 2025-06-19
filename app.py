from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from analyzer import predict_scratch_card
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

class GridRequest(BaseModel):
    grid: List[List[int]]
    target_num: Optional[int] = None
    iterations: Optional[int] = 10000

@app.post("/predict")
async def predict(req: GridRequest):
    result = predict_scratch_card(req.grid, req.target_num, req.iterations)
    return result