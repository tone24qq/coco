# Scratchcard Board Inference Service

## Overview
本服務專注於「刮卡盤面推理」：
- API 不做 OCR、不接圖片。
- GPT / 自訂 GPT 負責讀圖與人工確認盤面。
- API 僅接收結構化 board + target_number，輸出最佳位置與完整推理細節。

## GPT + API 兩段式流程
1. GPT 讀取圖片並解析盤面。
2. GPT 將未開格統一轉成 `-1`，把完整表格列給使用者確認。
3. 若看不清楚，先標記 uncertain cells 並要求使用者確認。
4. 使用者確認後，GPT 才呼叫 `POST /infer_target_position`。

## API
- `GET /health`
- `POST /infer_target_position`

### Request example
```json
{
  "board": [
    [1, -1, 3],
    [-1, 5, -1]
  ],
  "target_number": 4,
  "source": "gpt_image_parse",
  "parse_snapshot": {
    "raw_cells": [],
    "notes": ""
  }
}
```

### Response highlights
- `best_cell`
- `candidate_cells` (排序)
- `confidence_score`
- `reasoning`
- `module_contributions`

## Run
```bash
uvicorn src.api:app --reload
```

## OpenAPI for Custom GPT Action
請使用：
- `openapi/gpt_action_infer_target_position.json`

此 OpenAPI 明確要求 GPT 先做人為確認再呼叫 API。
