# analyzer.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict
from collections import Counter

# ↓ 这里把 servers URL 填成你在 Render 上看到的域名，不要带 /openapi.json
app = FastAPI(
    title="Coco Scratch-Card Analyzer",
    openapi_url="/openapi.json",
    servers=[{"url": "https://coco-3clu.onrender.com"}]
)

class AnalyzeRequest(BaseModel):
    cards: List[List[List[int]]]      # 13 张历史卡的二维矩阵列表
    new_card: List[List[int]] = None  # 可选：使用者提供的待分析新卡
    base: int                         # 基准号码
    targets: List[int]                # 目标号码列表

class AnalyzeResponse(BaseModel):
    offsets: Dict[int, List[Tuple[Tuple[int,int], int]]]
    recommendations: Dict[int, List[Dict]]
    tail_resonance: Dict[int,int]
    avg_adjacency_density: float

def find_pos(card, num):
    for r, row in enumerate(card):
        for c, v in enumerate(row):
            if v == num:
                return r, c
    raise ValueError(f"{num} not in card")

def compute_offset_freq(cards, base, target):
    cnt = Counter()
    for card in cards:
        try:
            r1, c1 = find_pos(card, base)
            r2, c2 = find_pos(card, target)
            cnt[(r2-r1, c2-c1)] += 1
        except ValueError:
            pass
    return cnt

def tail_resonance(cards, tail):
    hit = 0
    for card in cards:
        if sum(1 for row in card for v in row if v % 10 == tail) >= 2:
            hit += 1
    return hit

def adjacency_strength(card):
    h, w = len(card), len(card[0])
    total = h*(w-1) + w*(h-1)
    cnt = 0
    for row in card:
        cnt += sum(abs(row[i+1]-row[i])==1 for i in range(w-1))
    for c in range(w):
        cnt += sum(abs(card[r+1][c]-card[r][c])==1 for r in range(h-1))
    return cnt/total

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 1. 计算偏移热度
    offset_table = {
        t: compute_offset_freq(req.cards, req.base, t).most_common(5)
        for t in req.targets
    }
    # 2. 推荐新卡位置
    recs = {}
    if req.new_card:
        r0, c0 = find_pos(req.new_card, req.base)
        for t, off in offset_table.items():
            recs[t] = [
                {"pos": (r0+dr, c0+dc), "freq": freq}
                for (dr, dc), freq in off
                if 0 <= r0+dr < len(req.new_card) and 0 <= c0+dc < len(req.new_card[0])
            ]
    # 3. 尾数共鸣与邻号密度
    tail_info = {d: tail_resonance(req.cards, d) for d in range(10)}
    avg_adj = sum(adjacency_strength(c) for c in req.cards) / len(req.cards)
    return {
        "offsets": offset_table,
        "recommendations": recs,
        "tail_resonance": {d: cnt for d, cnt in tail_info.items() if cnt > len(req.cards)/2},
        "avg_adjacency_density": avg_adj
    }
