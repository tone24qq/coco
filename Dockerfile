FROM python:3.11-slim

# 1. 設定工作目錄
WORKDIR /app

# 2. 安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 複製整個專案（包含 samples/）
COPY . .

# 4. 列出 /app/samples 內容，確認檔案已經在 image 裡
RUN echo ">>> /app/samples 內容：" && ls -l /app/samples

# 5. 啟動 API 伺服器
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]