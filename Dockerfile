FROM python:3.11-slim

# 清華源加速 + pip優化
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

COPY requirements.txt .

# 依照你原本的做法，先安裝 PyTorch（如果 requirements.txt 已去除 torch）
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.1.0+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
