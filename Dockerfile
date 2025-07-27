FROM python:3.9-slim

# 1. 优化 pip 设置（切换镜像、延长超时、关缓存）
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

# 2. 先升级 pip / setuptools / wheel
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# 3. 安装 CPU-only 版 PyTorch，再装其余依赖
#    这里假设你在 requirements.txt 中把 torch 行去掉，改为下面单独安装：
RUN pip install torch==2.1.0+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# 4. 复制源码并启动
COPY . .
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-80}"]
