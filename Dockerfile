# ---------- build stage ----------
FROM python:3.11-slim AS builder

# 安裝系統編譯工具（僅必要的）
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製並安裝生產用依賴
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 複製程式碼（含 src、tests、scripts…）
COPY . .



# ---------- runtime stage ----------
FROM python:3.11-slim

WORKDIR /app

# 把 builder 的可執行檔也一起拷過來──uvicorn, orjson 等 console_scripts  
COPY --from=builder /usr/local/bin /usr/local/bin  
# 把所有安裝的套件複製過來  
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11

# 複製程式碼
COPY --from=builder /app /app

# 建一個非 root user，提高安全性，並改掉 /app 權限
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app

USER appuser

# 預設執行命令 (可被 Render 的 $PORT 覆寫)
ENV PORT=8000
EXPOSE $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]