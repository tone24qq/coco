# WinWin Bingo Predictor Service

最小可部署版本的 WinWin Bingo 預測服務（FastAPI）。

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## 本地啟動

```bash
uvicorn winwin_service.api:app --host 0.0.0.0 --port 8000 --app-dir src
```

## API

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

### `GET /predict`

```bash
curl -s http://127.0.0.1:8000/predict | python -m json.tool
```

成功回傳欄位：

- `target_period`
- `latest_period`
- `top3`
- `kill_zone`
- `metadata`

失敗（抓取/解析異常）時，服務會 `502` fail-fast，不會回傳假預測。

## Render 部署

本專案提供 `Procfile`：

```Procfile
web: uvicorn winwin_service.api:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir src
```

## 測試

```bash
pytest -q
```

## 實際回應範例

```json
{"target_period":115017505,"latest_period":115017504,"top3":[[1,24,64],[1,24,71],[1,28,71]],"kill_zone":[6,9,11,20,40,54],"metadata":{"analyzed_draws":50,"valid_pool_size":74,"total_combinations":64824,"qualified_combinations":31462,"min_score_threshold":60}}
```
