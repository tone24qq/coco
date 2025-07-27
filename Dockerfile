FROM python:3.9-slim

# 1. 系統層面安裝必要套件（若有需要編譯原生模組可打開）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
# && rm -rf /var/lib/apt/lists/*

# 2. 設定 pip 環境變數：關閉快取、延長超時、改用清華 PyPI 鏡像
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# 3. 先升級 pip / setuptools / wheel，確保穩定版本
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# 4. 安裝專案相依
RUN pip install -r requirements.txt

# 5. 複製原始碼並啟動服務
COPY . .
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-80}"]
