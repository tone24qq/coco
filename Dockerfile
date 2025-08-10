FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# 如果無需編譯原生套件，可省略這段 apt
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       libopenblas-dev liblapack-dev \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# 若 requirements.txt 已含 torch，改成只跑 pip install -r requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
 && pip install -r requirements.txt

COPY . .
EXPOSE 8000
ENV LOG_LEVEL=DEBUG

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]