FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ wget \
      libopenblas-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
 && pip install \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
 && pip install -r requirements.txt

COPY . .

ENV LOG_LEVEL=DEBUG

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]