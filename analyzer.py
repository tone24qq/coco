from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict
from collections import Counter
import traceback
import numpy as np

app = FastAPI()

class AnalyzeRequest(BaseModel):
    cards: List[List[List[int]]]
    new_card: List[List[int]] = None
    base: int
    targets: List[int]

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        results = {}
        for target in req.targets:
            results[target] = []

            if req.new_card is None:
                continue

            for r in range(len(req.new_card)):
                for c in range(len(req.new_card[0])):
                    if req.new_card[r][c] == -1:
                        score, reasons = evaluate_stability(req.new_card, r, c, target)
                        guide = (
                            "Py_promote" if score >= 0.85 else
                            "Py_neutral" if score >= 0.75 else
                            "Py_suggest_ignore"
                        )
                        results[target].append({
                            "pos": [r + 1, c + 1],
                            "score": round(score, 4),
                            "py_guide": f"{guide}:{'+'.join(str(r) for r in reasons) if reasons else '穩定'}"
                        })

            results[target] = sorted(results[target], key=lambda x: -x["score"])[:3]

        return {
            "status": "success",
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }