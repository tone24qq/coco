# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from analyzer import Analyzer  # 請確認 analyzer.py 在相同資料夾下

class AnalyzeRequest(BaseModel):
    grid: list[list[int]]   # 二維列表，隱藏格用 -1 表示
    target: int             # 要查找的號碼

app = FastAPI()
analyzer = Analyzer()

@app.post("/analyze")
def do_analyze(request: AnalyzeRequest):
    grid = request.grid
    target = request.target
    # 呼叫 Analyzer 的 analyze 方法，並傳入 grid 與 target
    result_scores = analyzer.analyze(grid, target)
    # 根據需要，你可以只回傳 Top 3 的位置，
    # 這裡示範回傳完整分數字典
    return {"scores": result_scores}

# 如果你希望只回傳 Top 3，請改用下面這段：
# from operator import itemgetter
# @app.post("/analyze")
# def do_analyze(request: AnalyzeRequest):
#     grid = request.grid
#     target = request.target
#     scores = analyzer.analyze(grid, target)
#     # 取出分數最高的前三個 (pos, score)
#     top3 = sorted(scores.items(), key=itemgetter(1), reverse=True)[:3]
#     # 只回傳位置 ID (pos)
#     top3_positions = [pos for pos, _ in top3]
#     return {"target": target, "top3_positions": top3_positions}