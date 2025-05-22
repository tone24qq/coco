
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import traceback

app = FastAPI()

class AnalyzeRequest(BaseModel):
    cards: List[List[List[int]]]
    new_card: List[List[int]] = None
    base: int
    targets: List[int]

error_positions = {(3, 2), (4, 4)}
hit_positions = {(2, 3), (4, 1)}
mask_zones = {(1, 2), (3, 4)}

def rule_A2_熱點中心(r, c, rows, cols):
    if 1 < r < rows - 2 and 1 < c < cols - 2:
        return 0.1, "熱區中心"
    return 0, ""

def rule_A5_鄰接密度(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    cnt = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
                cnt += 1
    score = cnt * 0.03
    return min(score, 0.15), f"鄰接密度{cnt}"

def rule_M5_段差(row_vals, col_vals, val):
    score = 0
    for v in row_vals + col_vals:
        if abs(v - val) == 10:
            score += 0.2
    return score, "段差±10" if score > 0 else ""

def rule_F7_段差鏈(grid, r, c, val):
    count = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if abs(grid[nr][nc] - val) == 10:
                count += 1
    score = min(count * 0.1, 0.2)
    return score, f"段差鏈x{count}" if count else ""

def rule_F6_共現格(r, c, rows, cols):
    if (r in [0, rows-1] and c in [0, cols-1]) or (r == rows//2 and c == cols//2):
        return 0.1, "對稱/角落/中心格"
    return 0, ""

def rule_F8_遮蔽解鎖(r, c):
    return (0.1, "遮蔽熱點") if (r + 1, c + 1) in mask_zones else (0, "")

def rule_R7_反例排除(r, c):
    return (-0.15, "反例格") if (r + 1, c + 1) in error_positions else (0, "")

def rule_F5_命中記憶(r, c):
    return (0.12, "樣卡命中") if (r + 1, c + 1) in hit_positions else (0, "")

rules = [
    ("A2", lambda g, r, c, t, rv, cv, rows, cols: rule_A2_熱點中心(r, c, rows, cols)),
    ("A5", lambda g, r, c, t, rv, cv, rows, cols: rule_A5_鄰接密度(g, r, c)),
    ("M5", lambda g, r, c, t, rv, cv, rows, cols: rule_M5_段差(rv, cv, t)),
    ("F7", lambda g, r, c, t, rv, cv, rows, cols: rule_F7_段差鏈(g, r, c, t)),
    ("F6", lambda g, r, c, t, rv, cv, rows, cols: rule_F6_共現格(r, c, rows, cols)),
    ("F8", lambda g, r, c, t, rv, cv, rows, cols: rule_F8_遮蔽解鎖(r, c)),
    ("R7", lambda g, r, c, t, rv, cv, rows, cols: rule_R7_反例排除(r, c)),
    ("F5", lambda g, r, c, t, rv, cv, rows, cols: rule_F5_命中記憶(r, c)),
]

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        results = {}
        rows = len(req.new_card)
        cols = len(req.new_card[0])

        if not req.targets:
            req.targets = [req.base]

        print(">>> 進入補格API，targets =", req.targets)

        for target in req.targets:
            results[target] = []

            for r in range(rows):
                for c in range(cols):
                    if req.new_card[r][c] != -1:
                        continue

                    row_vals = [v for j, v in enumerate(req.new_card[r]) if v != -1 and j != c]
                    col_vals = [req.new_card[i][c] for i in range(rows) if req.new_card[i][c] != -1 and i != r]

                    total_score = 1.0
                    reasons = []
                    module_scores = {}

                    for name, rule in rules:
                        score, reason = rule(req.new_card, r, c, target, row_vals, col_vals, rows, cols)
                        total_score += score
                        module_scores[name] = round(score, 4)
                        if reason:
                            reasons.append(f"{name}:{reason}")

                    total_score = round(min(total_score, 1.5), 4)
                    guide = (
                        "Py_promote" if total_score >= 1.2 else
                        "Py_neutral" if total_score >= 1.0 else
                        "Py_suggest_ignore"
                    )
                    results[target].append({
                        "pos": [r + 1, c + 1],
                        "score": total_score,
                        "py_guide": f"{guide}:{'+'.join(reasons) if reasons else '穩定'}",
                        "module_scores": module_scores
                    })

            top3 = results[target]
            if len(top3) >= 2:
                top_scores = [r['score'] for r in top3]
                if top_scores[0] - top_scores[1] < 0.01:
                    best = max(top3, key=lambda x: len([k for k, v in x['module_scores'].items() if v > 0]))
                    for r in top3:
                        if r != best:
                            r['py_guide'] = "Py_neutral:排序重排"
                    results[target] = [best] + [r for r in top3 if r != best][:2]

            results[target] = sorted(results[target], key=lambda x: -x["score"])[:3]

        return {"status": "success", "results": results}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }
