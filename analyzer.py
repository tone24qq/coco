from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict
from collections import Counter

app = FastAPI()

class AnalyzeRequest(BaseModel):
    cards: List[List[List[int]]]      # 歷史卡片矩陣列表
    new_card: List[List[int]] = None  # 可選，新卡矩陣
    base: int                         # 基準號碼，用於偏移計算
    targets: List[int]                # 目標號碼列表

class AnalyzeResponse(BaseModel):
    offsets: Dict[int, List[Tuple[Tuple[int,int], int]]]
    recommendations: Dict[int, List[Dict]]
    pos_recommendations: Dict[int, List[Dict]]  # 絕對位置TopN
    tail_resonance: Dict[int,int]
    avg_adjacency_density: float

# 尋找指定數字在卡片中的位置
def find_pos(card: List[List[int]], num: int) -> Tuple[int,int]:
    for r, row in enumerate(card):
        for c, v in enumerate(row):
            if v == num:
                return r, c
    raise ValueError(f"{num} not in card")

# 計算基準號碼與目標號碼的偏移頻率
def compute_offset_freq(cards: List[List[List[int]]], base: int, target: int) -> Counter:
    cnt = Counter()
    for card in cards:
        try:
            r1, c1 = find_pos(card, base)
            r2, c2 = find_pos(card, target)
            cnt[(r2-r1, c2-c1)] += 1
        except ValueError:
            pass
    return cnt

# ===== 新增：絕對位置統計 TopN =====
def compute_position_freq(cards: List[List[List[int]]], target: int, top_n: int = 3) -> List[Dict]:
    counter = Counter()
    for card in cards:
        for r, row in enumerate(card):
            for c, v in enumerate(row):
                if v == target:
                    counter[(r, c)] += 1
    return [
        {"row": r+1, "col": c+1, "count": cnt}
        for (r, c), cnt in counter.most_common(top_n)
    ] if counter else []

# 計算卡片中尾數共鳴
def tail_resonance(cards: List[List[List[int]]], tail: int) -> int:
    hit = 0
    for card in cards:
        if sum(1 for row in card for v in row if v % 10 == tail) >= 2:
            hit += 1
    return hit

# 計算相鄰號碼密度
def adjacency_strength(card: List[List[int]]) -> float:
    h, w = len(card), len(card[0])
    total = h*(w-1) + w*(h-1)
    cnt = 0
    for row in card:
        cnt += sum(abs(row[i+1] - row[i]) == 1 for i in range(w-1))
    for c in range(w):
        cnt += sum(abs(card[r+1][c] - card[r][c]) == 1 for r in range(h-1))
    return cnt / total

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    # 1. 計算偏移熱度 (top 5)
    offset_table = {
        t: compute_offset_freq(req.cards, req.base, t).most_common(5)
        for t in req.targets
    }

    # 2. 基於新卡推薦位置
    recs: Dict[int, List[Dict]] = {}
    if req.new_card is not None:
        r0, c0 = find_pos(req.new_card, req.base)
        for t, off in offset_table.items():
            recs[t] = [
                {"pos": (r0+dr, c0+dc), "freq": freq}
                for (dr, dc), freq in off
                if 0 <= r0+dr < len(req.new_card)
                and 0 <= c0+dc < len(req.new_card[0])
                and not req.new_card[r0+dr][c0+dc]
            ]

    # 3. 絕對位置TopN 建議
    pos_recs = {
        t: compute_position_freq(req.cards, t, top_n=3)
        for t in req.targets
    }

    # 4. 尾數共鳴與鄰號密度
    tail_info = {d: tail_resonance(req.cards, d) for d in range(10)}
    avg_adj = sum(adjacency_strength(c) for c in req.cards) / len(req.cards)

    return AnalyzeResponse(
        offsets=offset_table,
        recommendations=recs,
        pos_recommendations=pos_recs,
        tail_resonance={d:cnt for d,cnt in tail_info.items() if cnt > len(req.cards)/2},
        avg_adjacency_density=avg_adj
    )
