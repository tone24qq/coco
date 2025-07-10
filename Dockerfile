FROM python:3.11-slim

WORKDIR /app

# 1. 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 複製程式碼與樣本
COPY . .

# 3. ★ 建置階段就產生全局熱力圖
RUN python build_global_pos_freq.py -s samples -o out_npz

# 4. ★ 把巨型 boards 轉成 sample_stats，並移除 boards 檔
RUN python build_sample_stats.py -s samples -o samples --drop-boards

# 5. （非必要）列出確認
RUN echo "=== out_npz ===" && ls -l out_npz && \
    echo "=== samples ===" && ls -l samples

# 6. 啟動：用 $PORT，並在 startup 內載 priors（見下方 app.py）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]