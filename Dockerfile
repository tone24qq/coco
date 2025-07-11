# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码並預先生成 NPZ
COPY . .


FROM python:3.11-slim AS runner
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=builder /app/out_npz ./out_npz
COPY --from=builder /app/samples ./samples

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]