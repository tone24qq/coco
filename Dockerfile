# 使用 Debian slim，避免 Alpine(musl) 造成 PyTorch 無 wheels 與執行相容性問題
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# 系統相依套件（torch 需要基本建置工具；也讓 numpy/scipy 之類能順利安裝）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先複製 requirements.txt，獨立安裝可利用快取
COPY requirements.txt ./
RUN pip install --force-reinstall numpy
# 先安裝對齊版本的 PyTorch 三件組（CPU）
# 注意：使用官方 PyTorch CPU 索引，版本彼此對應：torch 2.1.2 / torchvision 0.16.2 / torchaudio 2.1.2
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
 && pip install -r requirements.txt

# 再把專案原始碼放進來
COPY . .
ENV LOG_LEVEL=DEBUG
# 轉換所有 jsonl -> json 並預先建立記憶快取
RUN for f in data_archives/*x*.jsonl; do \
    shape=$(basename "$f" .jsonl); \
    python - <<'PY'
import orjson, json, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = src.with_suffix('.json')
if not dst.exists():
    data = [orjson.loads(line) for line in src.open('rb')]
    json.dump(data, dst.open('w', encoding='utf-8'), ensure_ascii=False)
PY "$f"; \
    python - <<'PY'
from agents.memory_agent import build_memory_agent
import pathlib, sys
shape = tuple(map(int, sys.argv[1].split('x')))
json_path = pathlib.Path(f'data_archives/{sys.argv[1]}.json')
build_memory_agent(shape, json_path)
PY "$shape"; \
done
# 預設執行指令（依你的專案入口）
# 若採方案4，app 模組是 app:app；若改用 coco_service.main，請調整為 coco_service.main:app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]