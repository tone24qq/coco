
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

class ProposedValue(BaseModel):
    pos: List[int]  # [row, col]
    value: int
    score: float
    py_guide: str

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    try:
        # 僅執行排序與篩選
        sorted_results = sorted(req.proposed_values, key=lambda x: -x.score)[:3]
        return {
            "status": "success",
            "results": {
                sorted_results[0].value: [
                    {
                        "pos": r.pos,
                        "score": round(r.score, 4),
                        "py_guide": r.py_guide
                    } for r in sorted_results
                ]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
