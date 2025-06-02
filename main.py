from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import numpy as np
from analyzer import analyze_full_board

class GridInput(BaseModel):
    grid: list[list[int]]

app = FastAPI()

# 支援 GET "/" 回 200
@app.get("/")
async def root():
    return {"status": "Service is running"}

# 支援 HEAD "/" 回 200（可選）
@app.head("/")
async def root_head():
    return {}

@app.post("/analyze")
async def analyze(input: GridInput = Body(...)):
    grid = input.grid
    if grid is None:
        raise HTTPException(status_code=422, detail="No grid data provided.")
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) != 1:
        raise HTTPException(status_code=422, detail="Grid rows are not of equal length.")
    n_rows, n_cols = len(grid), len(grid[0]) if grid else 0
    if n_rows == 0 or n_cols == 0:
        raise HTTPException(status_code=422, detail="Grid must be non-empty 2D array.")
    if n_rows > 100 or n_cols > 100:
        raise HTTPException(status_code=422, detail=f"Grid size {n_rows}x{n_cols} exceeds 100x100 limit.")
    try:
        arr = np.array(grid, dtype=int)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid grid data type: {e}")
    if arr.ndim != 2:
        raise HTTPException(status_code=422, detail="Grid data is not a 2D matrix.")
    try:
        score_matrix = analyze_full_board(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    return {"score_matrix": score_matrix.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)