# 使用 Debian slim，避免 Alpine(musl) 造成 PyTorch 無 wheels 與執行相容性問題
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# 系統相依套件（torch 需要基本建置工具；也讓 numpy/scipy 之類能順利安裝）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先複製 requirements.txt，獨立安裝可利用快取
COPY requirements.txt ./
RUN pip install --force-reinstall numpy
# 先安裝對齊版本的 PyTorch 三件組（CPU）
# 注意：使用官方 PyTorch CPU 索引，版本彼此對應：torch 2.1.2 / torchvision 0.16.2 / torchaudio 2.1.2
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
 && pip install -r requirements.txt

# 再把專案原始碼放進來
COPY . .

# 預設執行指令（依你的專案入口）
# 若採方案4，app 模組是 app:app；若改用 coco_service.main，請調整為 coco_service.main:app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]