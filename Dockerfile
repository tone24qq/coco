FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "10000"]
