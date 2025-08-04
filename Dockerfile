# 1. 基礎鏡像
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# 2. 系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 安裝 Python 依賴（充分利用 layer cache）
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install --force-reinstall numpy \
 && pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
 && pip install -r requirements.txt

# 4. 複製生成腳本和原始資料
COPY build_memories.py convert_to_jsonl.py ./
COPY data_archives ./data_archives

# 5. （可選）把 .jsonl 轉成 .json
RUN python convert_to_jsonl.py

# 6. 生成 .npz 快取並打印中文日誌
ENV MEMORY_SAMPLE_LIMIT=1000
RUN python build_memories.py

# 7. 複製其餘原始碼
COPY . .

ENV LOG_LEVEL=DEBUG

# 8. 容器啟動
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
