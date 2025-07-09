FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼 + 熱力圖資料
COPY . .

# ✅ 已預先準備 out_npz/*.npz、samples/*.npz，不需再重產
# ❌ 以下兩行可移除避免浪費時間或意外覆蓋
# RUN python3 build_global_pos_freq.py -s samples -o out_npz
# RUN python3 precompute_heatmap.py

# 啟動 API 伺服器
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]