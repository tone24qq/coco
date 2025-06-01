from fastapi import FastAPI, Response
import logging
import asyncio
import time
import uuid
from datetime import datetime  # ✅ 補上這行

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# 全域 session 資訊
session_id = str(uuid.uuid4())
start_time = time.time()

@app.on_event("startup")
async def startup_event_minimal():
    logger.info(f"✅ App started at {datetime.now().isoformat()} | session_id={session_id}")
    asyncio.create_task(keep_alive_task())

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

@app.on_event("shutdown")
async def shutdown_event():
    uptime = int(time.time() - start_time)
    logger.warning(f"⚠️ Application is shutting down! | session_id={session_id} | Uptime before shutdown: {uptime}s")

@app.api_route("/", methods=["GET", "HEAD"])
async def minimal_root():
    logger.info(f"📡 Root / endpoint was hit! | session_id={session_id}")
    return {"message": "Minimal root is alive", "session_id": session_id}

@app.get("/healthz")
async def minimal_healthz():
    logger.info(f"❤️ Healthz /healthz endpoint was hit! | session_id={session_id}")
    return {"status": "ok from minimal", "session_id": session_id}

@app.get("/health")
async def minimal_health():
    logger.info(f"💙 /health endpoint was hit! | session_id={session_id}")
    return {"status": "healthy", "session_id": session_id}

@app.api_route("/healthz", methods=["HEAD"])
async def minimal_healthz_head():
    return Response(status_code=200)