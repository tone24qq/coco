# ---- Base image ----
FROM python:3.11-slim

# ---- Set timezone & environment ----
ENV TZ=Asia/Taipei \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_GLOB="checkpoints/met_*x*.pth"

# ---- Create working directory ----
WORKDIR /app

# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy project source code ----
COPY . .

# ---- Copy model checkpoints ----
COPY checkpoints/ ./checkpoints/

# ---- Expose FastAPI port ----
EXPOSE 8000

# ---- Run the FastAPI app ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
