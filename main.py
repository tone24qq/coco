from fastapi import FastAPI, Response
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

@app.on_event("startup")
async def startup_event_minimal():
    logger.info("✅ Minimal app started successfully!")

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

# 👉 根據需要可選：支援 HEAD /healthz 明確避免 405
@app.api_route("/healthz", methods=["HEAD"])
async def minimal_healthz_head():
    return Response(status_code=200)