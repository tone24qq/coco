from fastapi import FastAPI
from pydantic import BaseModel
from analyzer import analyze

# Define the input data model using Pydantic v2
class BoardInput(BaseModel):
    board: list[list[int]]

app = FastAPI()

# Health check endpoint
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# Analysis endpoint
@app.post("/analyze")
async def analyze_board(input: BoardInput):
    """Accepts a board with masked cells and returns predictions."""
    result = analyze(input.board)
    return result