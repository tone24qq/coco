from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict
import traceback
import numpy as np

app = FastAPI()

class AnalyzeRequest(BaseModel):
    cards: List[List[List[int]]]
    new_card: List[List[int]] = None
    base: int
    targets: List[int]

def evaluate_stability(grid, r, c, val):
    penalties = 0
    context = []

    new_grid = [row[:] for row in grid]
    new_grid[r][c] = val

    row_vals = [v for j, v in enumerate(new_grid[r]) if v != -1 and j != c]
    col_vals = [new_grid[i][c] for i in range(len(new_grid)) if new_grid[i][c] != -1 and i != r]

    if val in row_vals:
        penalties += 0.4
        context.append("同行重複")
    if val in col_vals:
        penalties += 0.4
        context.append("同列重複")

    diffs = []
    for seq in [row_vals + [val], col_vals + [val]]:
        seq = sorted(seq)
        if len(seq) > 1:
            diffs += [abs(seq[i+1] - seq[i]) for i in range(len(seq)-1)]

    if diffs:
        std_dev = np.std(diffs)
        penalties += min((std_dev / 6) ** 1.5, 0.3)
        if std_dev > 5:
            context.append("段差偏移")

    score = 1.0 - penalties
    if np.isnan(score) or np.isinf(score):
        score = 0.0

    return round(score, 4), context

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