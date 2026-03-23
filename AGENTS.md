# Agent Implementation Specs (System Blueprint)

此文件定義了模型的底層行為與特徵工程協議，任何模型迭代必須遵循以下規範：

### 1. 數據演進策略 (3K - 1M Scale)
- **3,000 筆：** 基準驗證期，重點在於 `Omission (遺漏值)` 特徵的有效性。
- **100,000 筆：** 加入 `Co-occurrence (共現)` 矩陣，啟動 Transformer Self-Attention。
- **1,000,000 筆：** 全量訓練，開啟 `Cyclic (週期性)` 特徵捕捉，並執行 Walk-forward 長期驗證。

### 2. 核心特徵邏輯 (The 4 Soul Features)
1. **Omission (遺漏回歸)：** 計算 `Current_Gap` 與 `Avg_Gap`，捕捉幾何分布下的反彈訊號。
2. **Co-occurrence (局部連動)：** 透過 Attention 矩陣計算 80 個候選項間的「吸引/排斥」權重。
3. **Momentum (動態動量)：** 5/20/100 局滾動頻率 Delta 值，識別號碼受熱趨勢。
4. **Cyclic (時間週期)：** 局數末位與開獎時段的 Positional Encoding。

### 3. 模型約束 (Constraints)
- **Architecture:** Encoder-only Transformer (Layers=3, d_model=128)。
- **File Size:** Hard limit < 100MB (僅儲存 State Dict)。
- **Output:** 必須保留 1-80 的完整分數鏈 (0.0 - 1.0)。

### 4. Top 3 去重演算法 (Diversity Protocol)
排序後提取 Top 20，針對 Top 3 執行以下篩選：
- **尾數去重：** 同一尾數 (n%10) 在 Top 3 中不得重複超過 1 個。
- **區間去重：** Top 3 必須儘量跨越 1-40 與 41-80 兩大區間。
- **鄰近去重：** 避免輸出如 (22, 23, 24) 這種連號組合。
## Engineering rules (must follow)

- Read the relevant existing code before editing. Never invent file paths, module names, config keys, artifact names, CLI args, or schemas.
- Prefer minimal diffs. Reuse existing utilities, config loaders, logging style, and folder layout.
- Preserve public contracts unless the task explicitly requires a breaking change.
- If a contract changes, update all call sites, loaders, tests, docs, and sample commands in the same task.
- Fail fast on missing models, missing columns, schema mismatch, invalid config, artifact/version mismatch, or unsupported runtime state.
- Do not silently add fallback behavior for broken artifacts or missing fields.
- Deterministic by default: fixed seeds, stable sorting, explicit tie-breakers, no hidden randomness in inference.
- Never use future data. No random split. No leakage from target issue into features, retrieval, normalization, or rerank.
- Keep configs externalized. Do not hard-code runtime paths or experiment-only constants into production code.
- Do not present ranking scores as probabilities unless they are explicitly calibrated and documented.
- Saved model artifacts must stay under 100MB. Save only compact metadata plus state_dict / required artifacts.

## Planning protocol

- If the task changes training, inference, feature schema, output schema, artifact contract, API contract, or touches more than 3 files, create or update `PLANS.md` before coding.
- `PLANS.md` must include: goal, touched files, invariants, risks, validation steps, rollback plan.
- Do not start implementation until the plan maps every changed module and every required test.

## Data and modeling invariants

- Each issue must produce exactly 80 candidate rows.
- Each issue must have exactly one group_id.
- The system must preserve the complete 1-80 score chain.
- Retrieval features may only use windows strictly earlier than the target issue.
- Training/validation must use walk-forward / time-series split only.
- Top 20 / Top 10 / Top 3 outputs must be reproducible from the same inputs and artifacts.
- Top 3 diversity is a post-ranking rerank rule, not a label shortcut.
- If architecture, feature definition, or output semantics change, bump the pipeline/model version and update metadata/loaders in the same task.

## Implementation protocol

- Before editing, inspect the current implementations for: feature building, dataset creation, training, prediction, config loading, artifact loading, and tests.
- Prefer small composable functions over large monolithic functions.
- Keep boundary validation near I/O edges.
- Add or update tests for every behavioral change.
- Do not leave dead code, unused config keys, or half-migrated code paths.
- If a faster path is added, keep the old path only when explicitly required and clearly version-gated.

## Validation requirements

- Do not claim success unless commands were actually run.
- Run the repository’s canonical lint, test, and prediction smoke commands before finishing.
- Minimum required checks:
  - [REPLACE_WITH_REAL_LINT_COMMAND]
  - [REPLACE_WITH_REAL_TEST_COMMAND]
  - [REPLACE_WITH_REAL_PREDICT_SMOKE_COMMAND]
- In the final handoff, always report:
  - files changed
  - commands run
  - pass/fail results
  - unresolved risks
  - whether contracts/artifacts changed

## Review guidelines

- Treat as P0/P1 if you see:
  - data leakage
  - future-data usage
  - schema drift
  - hidden randomness
  - silent fallback
  - broken output contract
  - artifact/version mismatch
  - model size > 100MB
  - inference latency regression against budget
  - missing fail-fast checks on required artifacts/columns