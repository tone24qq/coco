# coco

## API Mainline SOP (Vision-first)

正式主線以 API 自動解析為主，不要求使用者先提供 `rows/cols` 或 `manual-grid`。

### Step 1: 呼叫 API
- `POST /board/parse`：只上傳圖片（可選 `strict`）
- `POST /board/predict-number-position`：上傳圖片 + `query_number`（可選 `strict`）

### Step 2: 系統自動流程
1. 自動偵測 board 外框
2. 自動判斷 rows/cols（在已知票種 shape 中自動選擇最佳）
3. 自動切格 + OCR
4. 建立 `grid` / `numbers_all` / `value_to_position`
5. 產生 `board_bbox` / `cell_boxes` / overlay
6. 執行 contract 檢查，輸出低信心格與人工複核標記

### Step 3: 回傳重點
- `shape`, `grid`, `numbers_all`, `value_to_position`
- `query_number`, `query_status`, `exact_positions`, `top5_position_candidates`
- `overlay_image_base64`（若有產生 overlay）
- `parse_confidence`, `confidence_summary`, `low_confidence_cells`
