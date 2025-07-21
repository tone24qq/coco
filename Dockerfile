FROM python:3.11-slim
WORKDIR /app

# 1) 系統相依
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 2) Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) 專案程式碼
COPY . .

# 4) ⬇️ 直接 in-place redump 所有 *.pkl ⬇️
RUN python - <<'PY'
import pathlib, sys, joblib
from coco_common.scalers import Float32StandardScaler  # noqa: F401

# 把類別掛到舊路徑，舊 pickle 才能被讀進來
sys.modules['__main__'].Float32StandardScaler = Float32StandardScaler

for pkl in pathlib.Path("models").glob("*.pkl"):
    print(f"🔄 redump {pkl.name} (in-place)")
    mdl = joblib.load(pkl)
    joblib.dump(mdl, pkl)          # 覆寫同一檔名
PY

CMD ["uvicorn", "coco_service.main:app", "--host", "0.0.0.0", "--port", "8000"]