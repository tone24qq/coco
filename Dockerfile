# ---------- build stage ----------
FROM python:3.11-slim AS builder

# 1) 安裝系統相依套件（只挑必要）
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) 複製並安裝 Python 依賴
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 3) 複製整個專案（含 src / tests / scripts …）
COPY . .

# ---------- runtime stage ----------
FROM python:3.11-slim

# 建一個非 root user，安全一點
RUN adduser --disabled-password --gecos "" appuser
USER appuser
WORKDIR /app

# 從 builder 複製 site-packages 與專案程式
COPY --from=builder /usr/local/lib/python*/site-packages /usr/local/lib/python*/site-packages
COPY --from=builder /app .

# 預設執行命令 (可用環境變數覆蓋)
ENV PORT=8000
EXPOSE $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
