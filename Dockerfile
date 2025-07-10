FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# （可選）列清單
RUN echo "=== out_npz ===" && ls -l out_npz || true && \
    echo "=== samples ===" && ls -l samples || true

# 取代原來那行
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]