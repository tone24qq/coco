FROM python:3.11-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼與樣本資料
COPY . .

# 建立 out_npz/global_pos_freq_*x*.npz → 給 analyzer 模組使用
RUN python3 build_global_pos_freq.py -s samples -o out_npz

# 建立 samples/pos_freq_*x*.npz → 給 fallback 使用（例如只有一個 shape 時）
RUN python3 precompute_heatmap.py

# 啟動 API 伺服器
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]