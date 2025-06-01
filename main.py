from fastapi import FastAPI
import logging # 為了看到 Render 上的日誌

app = FastAPI()
logger = logging.getLogger("uvicorn.error") # 或 "uvicorn"

@app.on_event("startup")
async def startup_event_minimal():
    logger.info("Minimal app started successfully!")

@app.get("/")
async def minimal_root():
    logger.info("Minimal root / endpoint was hit!")
    return {"message": "Minimal root is alive"}

@app.get("/healthz") # Render 通常會檢查這個或類似路徑
async def minimal_healthz():
    logger.info("Minimal healthz /healthz endpoint was hit!")
    return {"status": "ok from minimal"}

# 如果您想在本機執行 (Render 會用自己的命令啟動 uvicorn)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)