print( >>>>已進入簡化版 <<<<<
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ProposedValue(BaseModel):
    pos: List[int]
    value: int
    score: float
    py_guide: str

class SimplifiedAnalyzeRequest(BaseModel):
    new_card: List[List[int]]
    proposed_values: List[ProposedValue]

@app.post("/analyze")
def analyze(req: SimplifiedAnalyzeRequest):
    try:
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
