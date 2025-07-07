FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies early so the layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

RUN python precompute_heatmap.py

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
