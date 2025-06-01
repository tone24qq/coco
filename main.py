from fastapi import FastAPI, Response
import logging
import asyncio
import time
import uuid

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# 全域 session 資訊
session_id = str(uuid.uuid4())
start_time = time.time()

# 👉 App 啟動時：印 log 並啟動防關閉的常駐任務
@app.on_event("startup")
async def startup_event_minimal():
    logger.info(f"✅ App started at {datetime.now().isoformat()} | session_id={session_id}")
    asyncio.create_task(keep_alive_task())  # 👈 長駐任務啟動

# 🧠 保持存活的任務，每 60 秒報平安一次並顯示 uptime 秒數
async def keep_alive_task():
    counter = 0
    while True:
        try:
            counter += 1
            uptime = int(time.time() - start_time)
            logger.info(f"💡 Still alive... Ping #{counter} | Uptime: {uptime}s | session_id={session_id}")
            await asyncio.sleep(60)
        except Exception as e:
            logger.exception(f"🔥 Keep-alive crashed: {e}")

# 🧼 當應用被關閉（不管是正常還是異常）時執行
@app.on_event("shutdown")
async def shutdown_event():
    uptime = int(time.time() - start_time)
    logger.warning(f"⚠️ Application is shutting down! | session_id={session_id} | Uptime before shutdown: {uptime}s")

# 👉 主路由：同時支援 GET 和 HEAD（避免 Render/UptimeRobot 405）
@app.api_route("/", methods=["GET", "HEAD"])
async def minimal_root():
    logger.info(f"📡 Root / endpoint was hit! | session_id={session_id}")
    return {"message": "Minimal root is alive", "session_id": session_id}

# 👉 健康檢查路由（Render 預設或手動可設為 /healthz）
@app.get("/healthz")
async def minimal_healthz():
    logger.info(f"❤️ Healthz /healthz endpoint was hit! | session_id={session_id}")
    return {"status": "ok from minimal", "session_id": session_id}

# 👉 備用健康檢查路徑（有些平台預設抓 /health）
@app.get("/health")
async def minimal_health():
    logger.info(f"💙 /health endpoint was hit! | session_id={session_id}")
    return {"status": "healthy", "session_id": session_id}

# 👉 支援 HEAD /healthz：避免 405 Method Not Allowed
@app.api_route("/healthz", methods=["HEAD"])
async def minimal_healthz_head():
    return Response(status_code=200)
