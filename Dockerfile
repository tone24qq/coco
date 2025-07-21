FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .

# 關鍵就這一行，直接加在 pip install 前
RUN apt-get update && apt-get install -y libgomp1

RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "coco_service.main:app", "--host", "0.0.0.0", "--port", "8000"]