# Auto-generated Dockerfile (CPU) — coco-16-36-50-thk-update-loader-and-dataset-for-target-support-2025-07-26.zip
    FROM python:3.11-slim AS runtime

    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libopenblas-dev \
        && rm -rf /var/lib/apt/lists/*

    ENV PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1

    # Use Tsinghua mirror by default; override with --build-arg PYPI_MIRROR=...
    ARG PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
    ENV PIP_INDEX_URL=${PYPI_MIRROR}

    WORKDIR /app

    COPY requirements.txt /app/requirements.txt

    ARG TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
    RUN pip install --upgrade pip setuptools wheel && \
        pip install --extra-index-url ${TORCH_EXTRA_INDEX_URL} -r requirements.txt || \
        (echo "requirements 安裝失敗，退回最小依賴…" && \
         pip install --extra-index-url ${TORCH_EXTRA_INDEX_URL} fastapi uvicorn[standard] numpy scikit-learn joblib pydantic)

    COPY . /app

    ENV MODELS_DIR=models \
        PRIOR_DIR=out_npz \
        SAMPLES_DIR=samples \
        BOARD_ROWS=8 \
        BOARD_COLS=10 \
        MAX_VALUE=80 \
        PORT=8000

    # Optional prewarm step; do not fail build.
    RUN python - <<'PY' || true
import os
print("[PREWARM] MODELS_DIR:", os.environ.get("MODELS_DIR"))
print("[PREWARM] PRIOR_DIR :", os.environ.get("PRIOR_DIR"))
print("[PREWARM] SAMPLES_DIR:", os.environ.get("SAMPLES_DIR"))
PY

    RUN useradd -m appuser
    USER appuser

    EXPOSE 8000

    HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
      CMD curl -fsS http://127.0.0.1:${PORT}/ || exit 1

    CMD uvicorn coco-16-36-50-thk-update-loader-and-dataset-for-target-support-2025-07-26.app:app --host 0.0.0.0 --port ${PORT} --workers 1
