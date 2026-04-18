# Local Multi-Size Training Pipeline（一條指令重訓）

## Python 版本建議
- 建議使用 **Python 3.11**（主線已驗證）。
- `scripts/run_training_pipeline.py` 對 Python 3.14+ 預設會 fail-fast。
- 若你要強制嘗試未驗證版本，可加：`--allow-unsupported-python`。

## 一條指令
```bash
python scripts/run_training_pipeline.py \
  --input-dir . \
  --generate-synthetic \
  --enable-inference
```

此命令會依序完成：
1. 掃描 `--input-dir` 下所有 `.xlsx`（支援多尺寸，優先從檔名抓 `10x12` 這類 hint，否則回退內容推斷）。
2. 驗證每個候選子矩陣是否為 `1..N` 的完整 permutation（缺值/重複/越界會 rejected 並寫進 audit）。
3. 產出多尺寸 corpus（每筆含 `rows/cols/size_class/board_size/source_file/sheet_name/board_id`）。
4. 建 ranking dataset。
5. split train/valid/holdout（保留 per-size 統計）。
6. 訓練 global 模型。
7. 依 `size_class` 訓練 per-size 模型（資料不足會跳過，runtime 用 global fallback）。
8. 寫出 model registry + readiness 報告。

## 主要輸出檔案
- `data/full_boards/full_board_corpus.jsonl`
- `reports/full_board_corpus_audit.json`
- `reports/multisize_corpus_summary.json`
- `data/ranking/ranking_dataset.parquet`
- `data/ranking/splits/train.parquet`
- `data/ranking/splits/valid.parquet`
- `data/ranking/splits/holdout.parquet`
- `data/ranking/splits/split_summary.json`
- `artifacts/global/main_ranker.pkl`
- `artifacts/sizes/<size_class>/main_ranker.pkl`
- `artifacts/model_registry.json`
- `reports/multisize_training_summary.json`
- `reports/runtime_readiness_report.json`

## Runtime 選模規則
- `src/main_ranker.py` 會先用盤面大小（`rows x cols`）組成 `size_class`。
- 若 registry 有對應 `per_size[size_class]` 且 artifact 存在，就用該模型。
- 否則 fallback 到 `global` 模型。
- 若 `strict_missing_artifact=true` 且 global/per-size 都不存在，會 fail-fast。

## 常用參數
- `--model-strategy auto|per_size|global_only`
- `--min-real-boards-per-size 5`
- `--mask-ratios 0.1,0.2,0.3,0.5`
- `--holdout-ratio 0.2`
- `--max-workers 1`
