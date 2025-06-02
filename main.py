import asyncio
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from analyzer import analyze

# Define the input data model using Pydantic v2
class BoardInput(BaseModel):
    board: list[list[int]]

app = FastAPI()

# ================== 健康检查接口 ==================
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# ================== 自我 Ping 后台任务 ==================
async def self_ping_task():
    """
    每 60 秒向本服务的 /healthz 发一次请求，保持服务活跃。
    """
    url = "http://localhost:10000/healthz"
    # 如果你的线上域名不是 localhost:10000，请改成相应的完整地址，
    # 例如 "https://your-app.onrender.com/healthz"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                await client.get(url)
            except Exception:
                # 忽略任何报错，继续等待下一次
                pass
            await asyncio.sleep(60)


@app.on_event("startup")
async def on_startup():
    """
    FastAPI 启动后立刻创建后台任务，让 self_ping_task 持续运行。
    """
    asyncio.create_task(self_ping_task())


# ================ 分析接口 ================
@app.post("/analyze")
async def analyze_board(input: BoardInput):
    """
    接受前端传来的 board，然后调用 analyzer.analyze 产出预测结果。
    """
    result = analyze(input.board)
    return result