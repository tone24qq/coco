# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 先把源码和样本都复制进来
COPY . .

# 通过环境变量决定要不要预生成 NPZ
# 默认 1：跳过预生成，构建瞬间完成
# 如果你真要生成，在 build 时传 --build-arg SKIP_PRECOMPUTE=0
ARG SKIP_PRECOMPUTE=1

RUN if [ "$SKIP_PRECOMPUTE" = "0" ]; then \
      echo ">>> 生成全局热力图和样本统计 NPZ"; \
      python3 build_global_pos_freq.py -s samples -o out_npz && \
      python3 precompute_heatmap.py; \
    else \
      echo ">>> 跳过 NPZ 预生成"; \
    fi

# （可选）验证一下目录
RUN echo "=== out_npz ===" && ls -l out_npz || true && \
    echo "=== samples ===" && ls -l samples || true

# 正式启动命令
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]