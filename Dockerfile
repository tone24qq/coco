# 1. 基础镜像
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# 2. 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ gfortran wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 安装 Python 依赖（利用 layer cache）
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
 && pip install --force-reinstall numpy \
 && pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
 && pip install -r requirements.txt

# 4. 复制业务代码和已生成的记忆库缓存（*.npz）
COPY . .
# 确保 data_archives 中已经存在 *_memory.npz 文件
# （在本地或 CI 中提前运行 `python build_memories.py` 生成并提交到仓库）

ENV LOG_LEVEL=DEBUG

# 5. 启动服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]