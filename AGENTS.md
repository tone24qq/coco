# AGENTS.md

## 目的
本專案禁止表面實作、空殼模組、未接線 helper、假報表、假 schema 完成。  
任何功能若未真正接入主線資料流 / 訓練流 / 回測流 / 預測流 / API 流，一律不得宣稱完成。

---

## 工作原則
1. 先找出本次需求會影響的主線入口。
2. 先列最小必要修改檔案，再開始改。
3. 嚴禁只做文件、設定、測試、schema、metadata 表面修改。
4. 每次提交前必須跑命令、看輸出、附證據。
5. 若做不到真正接入，必須明確回報「未完成」，不得包裝成已完成。

---

## 主線入口 / 主線共用核心
以下檔案視為主線入口或主線核心：

- `src/train.py`
- `src/predict.py`
- `src/backtest.py`
- `src/api.py`
- `src/artifacts.py`
- `src/build_features.py`
- `src/runtime_scoring.py`
- `src/modeling.py`

---

## Definition of Done
只有同時滿足以下條件，才可宣稱完成：

1. **必須修改至少一個主線入口或主線共用核心**
   - 若需求聲稱「接入主線」，但沒有改到上述檔案之一，視為未完成。

2. **新功能必須真的被主流程呼叫到**
   - 不接受孤立 helper。
   - 不接受新增模組但沒有 import / call path。
   - 不接受只有檔案存在、只有函式存在、只有 schema 存在。

3. **主線行為必須真的改變**
   - 至少要有一個可觀察差異，例如：
     - 輸出 JSON 有新增且真正來自主流程的欄位
     - 排序結果 / `final_score` / `top3` / metadata / artifacts / reports 有真實差異
     - feature contract / artifact contract / fail-fast 行為有真實差異

4. **不得只做表面層修改**
   - 不接受只改 `README.md / configs / tests / reports / response schema / metadata`
   - 不接受只補說明文字
   - 不接受只做假的報表輸出
   - 不接受只把需求寫進回覆，不落實到程式

5. **必須有可驗證產物或日誌**
   - 例如：
     - `models/lightgbm_ranker.txt`
     - `models/logistic_regression.pkl`
     - `models/feature_columns.json`
     - `models/metadata.json`
     - `reports/latest_prediction.json`
     - `reports/backtest_experiment_summary.json`
     - `reports/predictability_test.json`
   - 產物必須非空，且內容與本次修改有關。

6. **若未真正接入，必須明確回報未完成**
   - 不得用「已完成框架」「已預留接口」「已支援未來擴充」冒充完成。

---

## Forbidden Shortcuts
以下情況一律視為未完成：

- 只補 `README`
- 只補 config
- 只補 tests
- 只補 API schema
- 只補 response model
- 只補 metadata 欄位
- 只補 reports 輸出
- 只新增 helper module
- 只新增 explain / consensus / snapshot / audit 模組但主線沒接
- 只做假 ablation
- 只做假 consensus
- 只做假 explain
- 只做假 fail-fast
- 只改字串或欄位名稱讓表面看起來像有做
- 只在回覆裡聲稱已接入，但沒有程式證據
- 只建立空殼檔案或 placeholder 函式
- 只把結果寫進 json/csv，但訓練、預測、回測主路徑沒用到

---

## Required Workflow
每次任務都必須依序執行：

1. **先列本次主線入口**
   - 指出本次需求影響哪個主線入口：
     - train
     - predict
     - backtest
     - api
     - feature building
     - runtime scoring
     - artifacts
     - modeling

2. **列最小必要修改檔案**
   - 只列真正必要檔案。
   - 不可先大量新增無關檔案。

3. **實作主線接入**
   - 必須真的接到 call path。
   - 必須真的進入資料流 / 模型流 / scoring 流 / API 流。

4. **補對應測試**
   - 測試必須驗證「真的接上」，不是只驗 schema 存在。

5. **執行命令**
   - 沒執行過不得宣稱完成。

6. **檢查輸出**
   - 必須確認產物、報表、API、log、測試結果真的存在。

7. **最後回報**
   - 必須附上命令、測試、產物、未完成項目。

---

## 測試要求（Testing Standards）

每個 Agent / Codex 任務在宣稱完成前，必須先完成「主線接入驗證 + 測試驗證 + 證據輸出」。  
沒有實際命令、沒有實際輸出、沒有實際測試結果，一律不得宣稱完成。

### 基本靜態檢查
至少必須通過：
- `flake8 agent.py`

若本次修改涉及 `src/` 或 `tests/`，還必須通過：
- `flake8 src tests`

若專案有額外檢查工具，也必須一併執行：
- `python -m py_compile $(git ls-files '*.py')`
- `pytest -q`

### 單元測試
測試檔案命名範例：
- `tests/test_agents/test_<agent_name>.py`

若本次修改涉及主線，至少必須補齊下列其中一類測試：

1. **主線接線測試**
   - 驗證新功能真的被 train / predict / backtest / api 呼叫到

2. **效果差異測試**
   - 驗證功能開關前後，輸出真的不同
   - 禁止只測欄位存在

3. **合約測試**
   - 驗證 artifact contract / feature contract / ranking contract / fail-fast contract

4. **端到端測試**
   - 驗證從資料 -> feature -> ranking dataset -> train / predict 的主線可執行

### 必跑測試命令
- `pytest -q`

若本次修改涉及主線，還必須至少能指出本次直接相關、實際跑過的測試檔，例如：
- `pytest -q tests/test_agents/test_phase1_predict_schema.py`
- `pytest -q tests/test_agents/test_phase2_api_and_pipeline.py`

### 無證據不得宣稱完成
- 沒有實際執行命令輸出，不得寫「已測試通過」
- 沒有實際產物，不得寫「已接入完成」
- 沒有實際差異，不得寫「已實作」

---

## 驗收規則（Acceptance Gates）

### Gate 1：主線必改
若需求屬於以下任一情況，**必須改主線檔案**：

- 接主線
- 接 train
- 接 predict
- 接 backtest
- 接 api
- 接 artifacts
- 接 runtime scoring
- 接 retrieval
- 接 explain / snapshot / consensus / provenance / audit 到正式流程

若沒改主線檔案，直接視為未完成。

### Gate 2：禁止孤兒模組
新檔案若沒有被主線 import / call，視為無效實作。

### Gate 3：禁止假輸出
若新增 report / metadata / schema 欄位，但其值不是主流程真實算出，視為未完成。

### Gate 4：必須可觀察
至少要有一個可觀察成果：

- 測試通過
- 產物生成
- API 響應改變
- 排名結果改變
- fail-fast 生效
- 回測輸出改變

### Gate 5：未完成必須明講
若只做到部分，必須分開列：
- 已完成
- 未完成
- 風險 / 缺口

不得用模糊字眼混過。

---

## Mandatory Verification Gate

### 強制驗收腳本
每次新增功能、修改主線、補接線後，必須新增或更新一個可執行驗收腳本：

- `scripts/verify_mainline.sh`

若專案原本已有同類腳本，可沿用，但必須真的更新內容，不可只保留空殼。

### 驗收腳本用途
`verify_mainline.sh` 是完成宣稱前的強制門檻。  
沒有執行 `verify_mainline.sh`，一律不得宣稱完成。

### Hard Rule
只要本次新增了任何功能，但沒有同步更新 `scripts/verify_mainline.sh` 來驗證該功能，一律視為未完成，不得提交，不得宣稱完成。

### 驗收腳本最低要求
`verify_mainline.sh` 必須至少做到以下事情：

1. 啟用 fail-fast：
   - `set -euo pipefail`

2. 明確執行：
   - 靜態檢查
   - 單元測試
   - 至少一組主線流程驗證

3. 主線流程驗證至少覆蓋對應路徑：
   - 訓練改動：驗證 train 主線真的吃到新功能
   - 預測改動：驗證 predict 主線真的吃到新功能
   - 回測改動：驗證 backtest 主線真的吃到新功能
   - API 改動：驗證 `/health` 或 `/predict` 真實反映新功能
   - runtime scoring 改動：驗證分數鏈真的變動，而不是只新增欄位

4. 對每個新功能都必須有對應檢查：
   - 不是只看檔案存在
   - 不是只看 metadata 有欄位
   - 必須檢查主流程輸出、日誌、模型輸出、API 回傳、報表內容中至少一項真的改變

5. 驗收失敗時必須 exit non-zero，不可吞錯，不可繼續假裝完成。

---

## Feature-specific Verification Rules
每新增一種功能，都必須補一條「功能級驗收」，加進：
- `scripts/verify_mainline.sh`
- 對應的 `tests/test_agents/test_*.py`

沒有功能級驗收，視為未完成。

### 1. 若新增 retrieval / similarity window / retrieval feature
必須驗證：
- `predict.py` 或 `build_features.py` 主流程真的呼叫到 retrieval
- 輸出中真的有 retrieval 產生的分數或排序影響
- 不是只多一個 metadata 欄位

至少要有一個驗收檢查像這樣：
- 檢查 `ranking_score_table` 中 `retrieval_score` 非全 0
- 或檢查 `final_score` 真的受 `retrieval_score` 影響
- 或檢查 `retrieval_top_matches` 非空，且不是假資料

### 2. 若新增 consensus
必須驗證：
- 不是只產生 `source_consensus_report.json`
- 主流程對 mismatch / all failed / partial 有實際行為
- 若規格要求 fail-fast，就必須真的 fail-fast

至少要驗收：
- ok 情況
- mismatch 情況
- all failed 情況

### 3. 若新增 explain
必須驗證：
- explain 不是固定字串或純欄位搬運
- explain 內容至少引用真實預測輸出的一部分
- 若規格要求對 score chain 解釋，則輸出需含實際 score chain

### 4. 若新增 backtest 額外報表
必須驗證：
- 不是只生成空 json/csv
- 報表內容來自實際 backtest 計算
- 欄位值不是寫死常數或 placeholder

### 5. 若新增 train metadata / snapshot / provenance
必須驗證：
- 主流程真的在 train 或 predict 中讀寫
- 不接受只有檔案存在
- 要檢查 metadata / snapshot 的內容與實際資料一致

### 6. 若修改 runtime scoring
必須驗證：
- 新分數欄位進入 `final_score`
- 排序結果真的可能改變
- 不是只增加欄位但 `final_score` 未使用

---

## 建議驗收測試類型
若需求容易被做成空殼，優先補這些測試：

1. **Mainline wiring test**
   - 驗證新功能真的從主入口走到主流程

2. **Feature effect test**
   - 驗證功能開 / 關時，`final_score` / `top3` / `metadata` / `reports` 真的有差異

3. **Artifact contract test**
   - 驗證模型檔、特徵檔、metadata 檔存在且非空
   - 缺任何必要檔案應 fail-fast

4. **Fail-fast test**
   - 缺欄位、缺模型、contract 不符時，必須真的報錯
   - 不接受 silent fallback

5. **End-to-end contract test**
   - 從 processed history -> features -> ranking dataset -> train / predict 整段可跑

---

## Required Tests
測試檔案命名範例：
- `tests/test_agents/test_<agent_name>.py`

但只寫測試不算完成。  
測試必須對應主線行為，不接受只有 helper test。

每次新增功能至少要補：
1. 一個 fail-fast / contract test
2. 一個主流程接入 test
3. 一個輸出驗證 test

### 必備測試類型
至少覆蓋以下三類中的對應項：

#### A. Contract / Fail-fast
例如：
- 缺 artifact 時應 fail-fast
- feature columns mismatch 應 fail-fast
- consensus all failed 應 fail-fast
- group ranking contract 錯誤應 fail-fast

#### B. Mainline Integration
例如：
- `train.py` 跑過後產物真的包含新功能結果
- `predict.py` 跑過後輸出真的包含主流程使用結果
- `backtest.py` 跑過後 summary 真來自計算
- `api.py` 回應真反映主線狀態

#### C. Output Reality Check
例如：
- 分數不是全 0
- 報表不是空殼
- explain 不是固定模板
- consensus 對不同輸入有不同結果
- runtime scoring 真的改變排序

---

## Required verify_mainline.sh Contents
`scripts/verify_mainline.sh` 至少要包含下列結構，且需依本次任務擴充：

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] lint"
flake8 agent.py
flake8 src tests

echo "[2/6] compile"
python -m py_compile $(git ls-files '*.py')

echo "[3/6] unit tests"
pytest -q

echo "[4/6] targeted tests"
# 依本次任務新增，例如：
# pytest -q tests/test_agents/test_phase1_fail_fast.py
# pytest -q tests/test_agents/test_phase1_predict_schema.py
# pytest -q tests/test_agents/test_phase3_provenance_and_consensus.py

echo "[5/6] mainline execution"
# 依本次任務執行最小主線驗證，例如：
# python -m src.train --config configs/train.yaml --input data/feature_store/ranking_dataset.csv
# python -m src.backtest --config configs/train.yaml --input data/feature_store/ranking_dataset.csv
# python -m src.predict --config configs/predict.yaml --output reports/latest_prediction.json

echo "[6/6] output assertions"
# 不可只檢查檔案存在，必須檢查內容真的有效
# 例如：
# python - <<'PY'
# import json
# from pathlib import Path
# p = Path("reports/latest_prediction.json")
# assert p.exists(), "missing prediction output"
# data = json.loads(p.read_text(encoding="utf-8"))
# assert len(data["ranking_score_table"]) == 80
# assert any(row["retrieval_score"] != 0 for row in data["ranking_score_table"])
# assert len(data["top20_numbers"]) == 20
# PY

echo "verify_mainline.sh PASSED"
重要限制
	•	不可把 verify_mainline.sh 做成只檢查檔名存在
	•	不可只 echo success
	•	不可只跑 pytest -q 就算驗收完成
	•	不可省略內容檢查

⸻

Required Final Report

最後回覆必須固定包含以下 7 項，缺一不可：
	1.	本次判定的主線入口檔案
	2.	真正修改的核心檔案
	3.	每個核心檔案的實際接入內容
	4.	尚未完成項目
	5.	實際執行命令
	6.	實際通過的測試
	7.	可驗證輸出檔案或日誌

⸻

Final Report Format

請嚴格使用以下格式：

1. 主線入口
	•	...

2. 核心修改檔案
	•	...

3. 實際接入內容
	•	檔案 -> 接了什麼主流程

4. 尚未完成
	•	...

5. 實際執行命令
6. 實際通過的測試
	•	...

7. 可驗證輸出
	•	...

⸻

回覆禁語

若沒有證據，禁止使用以下說法：
	•	已完成
	•	已接入主線
	•	已驗證
	•	已通過測試
	•	可直接上線
	•	已完整支援

除非後面有對應命令、測試、產物證據。

⸻

簡化判定規則

出現以下任一情況，直接判定為未完成：
	•	沒改主線檔案
	•	沒有真正 call path
	•	沒跑命令
	•	沒測試證據
	•	沒產物
	•	只有 README / config / tests / schema / metadata 修改
	•	只有新模組，沒有主線接入
	•	只有描述，沒有落地實作