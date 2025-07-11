FROM python:3.11-slim
WORKDIR /app

# 安裝相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼與原始樣本
COPY . .

# 在 build 時就預先產生 NPZ 檔
RUN python3 build_global_pos_freq.py -s samples -o out_npz \
 && python3 precompute_heatmap.py

# （可選）檢查生成結果
RUN echo "=== out_npz ===" && ls -l out_npz || true && \
    echo "=== samples ===" && ls -l samples || true

# 啟動指令：容器啟動後只載入已生成的 NPZ，並立刻開啟 HTTP 端口
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]