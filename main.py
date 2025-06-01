from fastapi import FastAPI, Response
import logging
import asyncio

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# 👉 App 啟動時：印 log 並啟動防關閉的常駐任務
@app.on_event("startup")
async def startup_event_minimal():
    logger.info("✅ Minimal app started successfully!")
    asyncio.create_task(keep_alive_task())  # 👈 長駐任務啟動

# 🧠 保持存活的任務，每 60 秒報平安一次
async def keep_alive_task():
    while True:
        logger.info("💡 Still alive... (保持活躍避免 Render 認為 idle)")
        await asyncio.sleep(60)

# 🧼 當應用被關閉（不管是正常還是異常）時執行
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("⚠️ Application is shutting down. This may be expected or an issue.")

# 👉 主路由：同時支援 GET 和 HEAD（避免 Render/UptimeRobot 405）
@app.api_route("/", methods=["GET", "HEAD"])
async def minimal_root():
    logger.info("📡 Root / endpoint was hit!")
    return {"message": "Minimal root is alive"}

# 👉 健康檢查路由（Render 預設或手動可設為 /healthz）
@app.get("/healthz")
async def minimal_healthz():
    logger.info("❤️ Healthz /healthz endpoint was hit!")
    return {"status": "ok from minimal"}

# 👉 備用健康檢查路徑（有些平台預設抓 /health）
@app.get("/health")
async def minimal_health():
    logger.info("💙 /health endpoint was hit!")
    return {"status": "healthy"}

# 👉 支援 HEAD /healthz：避免 405 Method Not Allowed
@app.api_route("/healthz", methods=["HEAD"])
async def minimal_healthz_head():
    return Response(status_code=200)
