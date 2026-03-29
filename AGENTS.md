# AGENTS.md

## Core Objective
This repository is not considered successful unless final holdout `same_triplet_2hit_rate >= 0.50`.
This project must optimize for a stricter production target:

**Primary KPI = `same_triplet_2hit_rate`**

Definition:
- For each draw, the system outputs 3 triplets as `top3`.
- Let `hit_count(triplet)` = overlap count between that triplet and the actual 20 drawn numbers.
- For that draw:
  - `same_triplet_2hit = 1` if **any one** of the 3 triplets has `hit_count >= 2`
  - otherwise `same_triplet_2hit = 0`
- Over an evaluation window:
  - `same_triplet_2hit_rate = mean(same_triplet_2hit)`

Business target:
- The official target is:
  - **at least 50% of draws must have one triplet hitting 2 or more numbers**
- In plain words:
  - `top3` 裡面，至少有同一組 triplet 命中 2 個以上的比率，要 >= 50%

---

## Metric Priority

Use metrics in this priority order:

1. `same_triplet_2hit_rate`  ← primary optimization target
2. `top1_2hit_rate`
3. `exact_hit@3`
4. `top3_at_least_one_hit_rate`

Required extra definitions:
- `top1_2hit_rate`:
  - first triplet only
  - 1 if `hit_count(top1) >= 2`, else 0
- `same_triplet_3hit_rate`:
  - 1 if any triplet has `hit_count >= 3`, else 0
- Existing metrics such as:
  - `top3_at_least_one_hit_rate`
  - `exact_hit@3`
  - `exact_hit@10`
  - `exact_hit@20`
  - adjacency metrics
  are **secondary diagnostics only**, not the main success criterion.

Do not declare success using only `top3_at_least_one_hit_rate`.

---

## Evaluation Rules

All evaluation must follow these rules:

- strict time-ordered walk-forward only
- no random split
- no future leakage
- untouched final holdout is mandatory
- final holdout must be split into at least 2 chronological blocks
- every block must be reported separately

Primary pass condition:
- final holdout `same_triplet_2hit_rate >= 0.50`

Block guardrail:
- each final holdout block must satisfy:
  - `same_triplet_2hit_rate >= 0.50`

Secondary guardrails:
- `top1_2hit_rate` must be reported
- must compare against simple baselines:
  - uniform random
  - frequency
  - previous neighbor
  - shift_m1
  - shift_p1
- chosen config must not be justified only by search-window performance

If final holdout fails the above target, mark result as:
- `passed = false`

---

## Production Contract

The live `/predict` endpoint must use the **same selected config** that was validated by walk-forward / final holdout.

Do not leave production on `DEFAULT_CONFIG` if reports are generated using another config.

Required rule:
- evaluation config
- saved best config
- live inference config

must be explicitly connected and traceable.

If the validated config is not wired into live prediction, treat the task as incomplete.

---

## Diversity Constraint

Do not game `same_triplet_2hit_rate` by making the 3 triplets near-duplicates.

Keep top3 diversified:
- avoid duplicate triplets
- avoid heavily overlapping triplets
- keep current diversified selection logic or make it stricter
- do not sacrifice diversity only to inflate one metric artificially

Preferred rule:
- shared numbers between selected triplets should be minimal
- fallback overlap relaxation is allowed only when necessary

---

## Required Report Outputs

The current branch already uses walk-forward and final-holdout scripts.
Extend the existing evaluation pipeline instead of creating a disconnected parallel metric path.

Required outputs must include the new KPI:

### `reports/walkforward/summary_report.json`
Must include:
- `same_triplet_2hit_rate`
- `top1_2hit_rate`
- `same_triplet_3hit_rate`

### `reports/final_holdout/summary_report.json`
Must include:
- `final_same_triplet_2hit_rate`
- `final_top1_2hit_rate`
- `final_same_triplet_3hit_rate`
- per-block versions of the same metrics
- pass/fail result based on `same_triplet_2hit_rate >= 0.50`

### `per_draw_report.csv`
Each draw must include:
- whether any triplet hit >= 2
- whether top1 hit >= 2
- whether any triplet hit >= 3

### `ablation_report.csv`
Must compare:
- old objective
- new objective
- baseline
- chosen config
- final holdout result

---

## Implementation Guidance

Prefer modifying the existing files in this branch:

- `scripts/strict_walkforward_search.py`
- `scripts/final_holdout_validation.py`
- `src/winwin_service/scoring.py`
- `src/winwin_service/api.py`

Required changes:
- add `same_triplet_2hit_rate` metric computation
- make search / ranking / selection prioritize this metric
- update summary outputs
- update pass/fail logic
- wire chosen config into live `/predict`

---

## Non-Negotiable Rules

- No future leakage
- No random split
- No claiming predictive certainty
- No reporting success from search-window metrics alone
- No production deployment using unvalidated config
- No “passed” status unless final holdout meets the 50% target

All outputs must be described as:
- scoring result
- ranking result
- backtest result
not certainty, not guarantee, not winning claim.