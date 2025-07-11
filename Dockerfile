FROM python:3.11-slim

# 这一行不动时，后面每次构建都会重用缓存  
# 加上下面这一行就能每次手动给它一个新值，强制刷新
ARG CACHEBUST=1

WORKDIR /app

# 先装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再把整个项目源码复制进来——只要 CACHEBUST 改了，这一层就会重跑
COPY . .

# Build 阶段就预先生成 NPZ
RUN python3 build_global_pos_freq.py -s samples -o out_npz \
 && python3 precompute_heatmap.py

# （可选）看一下到底有哪些 NPZ
RUN echo "=== out_npz ===" && ls -l out_npz || true && \
    echo "=== samples ===" && ls -l samples || true

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]