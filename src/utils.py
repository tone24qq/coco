from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_STORE_DIR = ROOT / "data" / "feature_store"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "configs"

RAW_FILES: list[str] = []
ZONE_NAMES = ["A", "B", "C", "D"]
V3_CORE20_COLUMNS = [
    "issue_zone_entropy",
    "issue_span_z50",
    "issue_sum_z50",
    "issue_consecutive_z50",
    "num_zone",
    "cand_repeat_last_draw",
    "cand_freq_smooth_20",
    "cand_freq_smooth_200",
    "cand_freq_trend_20_200",
    "cand_freq_ewma_hl50",
    "cand_gap_log1p",
    "cand_recency_ratio_200",
    "cand_gap_mad_5",
    "cand_recent_hit_decay_hl5",
    "cand_neighbor_pm1_decay_hl10",
    "cand_neighbor_pm2_decay_hl10",
    "cand_distance_kernel_last_draw_tau2",
    "cand_pmi_last_draw_sum_200",
    "cand_pmi_last_draw_max_200",
    "cand_handoff_pm1_lift_200",
    "ctx_prev_size_is_big",
    "ctx_prev_odd_even_is_odd",
    "ctx_size_big_ratio_w20",
    "ctx_odd_ratio_w20",
    "ctx_size_switches_w20",
    "ctx_odd_even_switches_w20",
    "cand_min_abs_distance_to_prev_draw",
    "cand_mean_abs_distance_to_prev_draw",
    "cand_is_exact_in_prev_draw",
    "cand_has_prev_pm1",
    "cand_has_prev_pm2",
    "cand_has_prev_pm3",
    "cand_count_prev_within_1",
    "cand_count_prev_within_2",
    "cand_count_prev_within_3",
    "cand_min_distance_to_recent_3",
    "cand_min_distance_to_recent_5",
    "cand_min_distance_to_recent_10",
    "cand_weighted_distance_decay_recent_10",
    "cand_count_recent_within_1_10",
    "cand_hits_last_200",
    "cand_hits_last_500",
    "cand_hits_last_1000",
    "cand_total_hits_all_time",
    "cand_current_gap_all",
    "cand_avg_gap_all",
    "cand_max_gap_all",
    "cand_today_hits",
    "cand_carryover_from_prev",
    "cand_pm1_neighbor_hits",
    "cand_pm2_neighbor_hits",
]

LEGACY_V3_COLUMNS = V3_CORE20_COLUMNS[:40]

CASCADE_V1_STAGE1_COLUMNS = [
    "cand_freq_smooth_20",
    "cand_freq_smooth_200",
    "cand_freq_trend_20_200",
    "cand_freq_ewma_hl50",
    "cand_gap_log1p",
    "cand_recent_hit_decay_hl5",
    "cand_neighbor_pm1_decay_hl10",
    "cand_neighbor_pm2_decay_hl10",
    "cand_distance_kernel_last_draw_tau2",
    "cand_pmi_last_draw_sum_200",
    "cand_handoff_pm1_lift_200",
]

CASCADE_V1_STAGE2_COLUMNS = [
    "issue_zone_entropy",
    "issue_span_z50",
    "issue_sum_z50",
    "issue_consecutive_z50",
    "num_zone",
    "cand_repeat_last_draw",
    "cand_pmi_last_draw_sum_200",
    "cand_pmi_last_draw_max_200",
    "cand_count_prev_within_1",
    "cand_count_prev_within_2",
    "cand_count_prev_within_3",
    "cand_count_recent_within_1_10",
]

CASCADE_V1_STAGE3_COLUMNS = [
    "number",
    "zone_id",
    "tail",
    "stage2_score",
    "stage2_rank",
]

PIPELINE_FEATURE_SCHEMAS = {
    "baseline_flat_score": {
        "flat": V3_CORE20_COLUMNS,
    },
    "cascade_v1": {
        "stage1": CASCADE_V1_STAGE1_COLUMNS,
        "stage2": CASCADE_V1_STAGE2_COLUMNS,
        "stage3": CASCADE_V1_STAGE3_COLUMNS,
    },
    "cascade_v2": {
        "stage1": CASCADE_V1_STAGE1_COLUMNS,
        "stage2": CASCADE_V1_STAGE2_COLUMNS,
        "stage3": CASCADE_V1_STAGE3_COLUMNS,
    },
}
LOGGER = logging.getLogger(__name__)


def normalize_feature_version(feature_version: str | None) -> str:
    v = str(feature_version or "v3_core20").strip()
    if v != "v3_core20":
        raise ValueError(
            f"unsupported feature_version: {v}; only v3_core20 is supported"
        )
    return v


def normalize_pipeline_version(pipeline_version: str | None) -> str:
    v = str(pipeline_version or "baseline_flat_score").strip()
    if v not in PIPELINE_FEATURE_SCHEMAS:
        raise ValueError(f"unsupported pipeline_version: {v}")
    return v


def get_pipeline_feature_schema(pipeline_version: str | None) -> dict[str, list[str]]:
    v = normalize_pipeline_version(pipeline_version)
    schema = PIPELINE_FEATURE_SCHEMAS[v]
    return {k: list(cols) for k, cols in schema.items()}


def validate_stage_feature_contract(
    pipeline_version: str,
    stage_name: str,
    columns: Sequence[str],
) -> None:
    schema = get_pipeline_feature_schema(pipeline_version)
    if stage_name not in schema:
        raise ValueError(f"pipeline={pipeline_version} has no stage={stage_name}")
    expected = list(schema[stage_name])
    actual = list(columns)
    if expected != actual:
        raise ValueError(
            f"pipeline={pipeline_version} stage={stage_name} schema mismatch"
        )


def min_required_history(
    feature_version: str, runtime_config: dict | None = None
) -> int:
    normalize_feature_version(feature_version)
    cfg = runtime_config or {}
    windows = cfg.get("core_windows", {})
    freq_long = int(windows.get("freq_long", 20))
    pmi_window = int(windows.get("pmi_window", 20))
    handoff_window = int(windows.get("handoff_window", 20))
    return max(freq_long, pmi_window, handoff_window) + 1


def classify_feature_mode(history_length: int) -> str:
    if history_length <= 20:
        return "short"
    if history_length <= 60:
        return "medium"
    if history_length <= 200:
        return "long"
    return "full"


def resolve_effective_windows(
    actual_history: int, runtime_config: dict | None = None
) -> dict[str, int]:
    cfg = runtime_config or _load_feature_runtime_config()
    windows = dict(cfg.get("core_windows", {}))
    resolved: dict[str, int] = {}
    for key, val in windows.items():
        w = int(val)
        if w <= 0:
            resolved[key] = 0
        else:
            resolved[key] = min(actual_history, w)
    return resolved


@dataclass
class TrainConfig:
    feature_min_history: int = 20
    backtest_splits: int = 5
    catboost_params: Dict[str, float] | None = None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        FEATURE_STORE_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def parse_numbers(row: pd.Series) -> List[int]:
    cols = [f"獎號{i}" for i in range(1, 21)]
    if not all(c in row.index for c in cols):
        cols = [f"n{i}" for i in range(1, 21)]
    nums = [int(row[c]) for c in cols]
    nums = sorted(nums)
    if len(nums) != 20:
        raise ValueError("Each draw must have 20 numbers")
    return nums


def zone_of(num: int) -> str:
    if num <= 20:
        return "A"
    if num <= 40:
        return "B"
    if num <= 60:
        return "C"
    return "D"


def classify_board(zone_counts: Dict[str, int]) -> str:
    values = [zone_counts[z] for z in ZONE_NAMES]
    if max(values) - min(values) <= 1:
        return "balanced"
    s = sorted(values, reverse=True)
    if s[0] >= 9:
        return "single_zone_burst"
    if s[0] >= 7 and s[1] >= 6 and s[2] <= 4:
        return "double_zone_shake"
    return "extreme_burst"


def _tail_features(numbers: Sequence[int]) -> Dict[str, float]:
    tails = [n % 10 for n in numbers]
    values, counts = np.unique(tails, return_counts=True)
    counter = {int(v): int(c) for v, c in zip(values, counts)}
    data = {f"tail_{i}_cnt": counter.get(i, 0) for i in range(10)}
    data["tail_unique_cnt"] = float(len(values))
    data["tail_max_cnt"] = float(max(counter.values()))
    data["same_tail_pair_cnt"] = float(sum(c * (c - 1) / 2 for c in counter.values()))
    return data


def _gap_features(numbers: Sequence[int]) -> Dict[str, float]:
    arr = np.array(sorted(numbers))
    gaps = np.diff(arr)
    consecutive_pairs = int((gaps == 1).sum())
    run = 1
    max_run = 1
    for g in gaps:
        if g == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return {
        "consecutive_pairs": float(consecutive_pairs),
        "max_consecutive_run": float(max_run),
        "run_len_ge_3": float(max_run >= 3),
        "run_len_ge_4": float(max_run >= 4),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std()),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "gap_lt_3_cnt": float((gaps < 3).sum()),
        "gap_ge_8_cnt": float((gaps >= 8).sum()),
    }


def _zone_counts(numbers: Sequence[int]) -> Dict[str, int]:
    return {
        "A": int(sum(1 for n in numbers if n <= 20)),
        "B": int(sum(1 for n in numbers if 21 <= n <= 40)),
        "C": int(sum(1 for n in numbers if 41 <= n <= 60)),
        "D": int(sum(1 for n in numbers if 61 <= n <= 80)),
    }


def _recent_frequency(history: Sequence[Sequence[int]], window: int) -> Dict[int, int]:
    draws = history[-window:]
    freq = {n: 0 for n in range(1, 81)}
    for d in draws:
        for n in d:
            freq[n] += 1
    return freq


def _shifted_set(numbers: Sequence[int], shift: int) -> set[int]:
    out = set()
    for n in numbers:
        s = n + shift
        if 1 <= s <= 80:
            out.add(s)
    return out


def _entropy(ps: Sequence[float], eps: float = 1e-12) -> float:
    return float(-sum(float(p) * np.log(float(p) + eps) for p in ps))


def _zscore(
    value: float, series: Sequence[float], window: int, eps: float = 1e-9
) -> float:
    recent = np.asarray(series[-window:], dtype=float)
    if recent.size < window:
        return 0.0
    std = float(recent.std(ddof=0))
    return float((value - float(recent.mean())) / (std + eps))


def _half_life_alpha(half_life: float) -> float:
    return float(1 - np.exp(np.log(0.5) / float(half_life)))


def _laplace_rate(c: float, window: int, alpha: float) -> float:
    return float((float(c) + alpha) / (float(window) + 2.0 * alpha))


def _exp_decay_weights(k: int, half_life: float) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=float)
    steps = np.arange(1, k + 1, dtype=float)
    return np.power(0.5, steps / float(half_life))


def _load_feature_runtime_config() -> dict:
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    runtime = {
        "feature_version": normalize_feature_version(
            cfg.get("feature_version", "v3_core20")
        ),
        "core_windows": cfg.get(
            "core_windows",
            {
                "z_window": 50,
                "freq_short": 20,
                "freq_long": 200,
                "pmi_window": 200,
                "handoff_window": 200,
            },
        ),
        "smoothing_alpha": float(cfg.get("smoothing_alpha", 0.5)),
        "decay_half_lives": cfg.get(
            "decay_half_lives", {"ewma": 50, "recent_hit": 5, "neighbor": 10}
        ),
        "distance_kernel_tau": float(cfg.get("distance_kernel_tau", 2)),
    }
    if os.getenv("FEATURE_RUNTIME_CONFIG_JSON"):
        runtime.update(json.loads(os.getenv("FEATURE_RUNTIME_CONFIG_JSON", "{}")))
    if os.getenv("FEATURE_VERSION_OVERRIDE"):
        runtime["feature_version"] = normalize_feature_version(
            os.getenv("FEATURE_VERSION_OVERRIDE")
        )
    return runtime


def _feature_version() -> str:
    return normalize_feature_version(_load_feature_runtime_config()["feature_version"])


def validate_feature_columns_contract(
    feature_columns: Sequence[str],
    feature_version: str,
    allow_legacy_subset: bool = False,
) -> None:
    normalize_feature_version(feature_version)
    cols = list(feature_columns)
    if cols == V3_CORE20_COLUMNS:
        return
    if allow_legacy_subset and cols == LEGACY_V3_COLUMNS:
        LOGGER.warning(
            "using legacy feature columns subset (len=%s < latest=%s)",
            len(cols),
            len(V3_CORE20_COLUMNS),
        )
        return
    if len(cols) != len(V3_CORE20_COLUMNS):
        raise ValueError(
            f"v3_core20 feature columns must be {len(V3_CORE20_COLUMNS)}, got {len(cols)}"
        )
    raise ValueError("v3_core20 feature columns must match fixed core20 order")


def _build_issue_features_v3(
    df: pd.DataFrame,
    min_history: int = 20,
    include_latest_for_inference: bool = False,
) -> pd.DataFrame:
    draws = [sorted(json.loads(v) if isinstance(v, str) else v) for v in df["numbers"]]

    def _derive_size_label(numbers: Sequence[int]) -> str:
        big = sum(1 for n in numbers if n >= 41)
        return "大" if big >= (20 - big) else "小"

    def _derive_odd_even_label(numbers: Sequence[int]) -> str:
        odd = sum(1 for n in numbers if n % 2 == 1)
        return "單" if odd >= (20 - odd) else "雙"

    size_series: list[str] = []
    odd_even_series: list[str] = []
    for i, nums in enumerate(draws):
        raw_size = str(df.iloc[i].get("size_label", "") or "").strip()
        raw_oe = str(df.iloc[i].get("odd_even_label", "") or "").strip()
        size_series.append(
            raw_size if raw_size in {"大", "小"} else _derive_size_label(nums)
        )
        odd_even_series.append(
            raw_oe if raw_oe in {"單", "雙"} else _derive_odd_even_label(nums)
        )

    rows: list[dict] = []
    z_window = int(_load_feature_runtime_config()["core_windows"].get("z_window", 50))
    end = len(df) if include_latest_for_inference else len(df) - 1

    spans: list[float] = []
    sums: list[float] = []
    consecutives: list[float] = []
    for i in range(min_history, end):
        nums = draws[i]
        arr = np.array(nums)
        zc = _zone_counts(nums)
        span = float(arr.max() - arr.min())
        total = float(arr.sum())
        consecutive_pairs = float((np.diff(np.array(nums)) == 1).sum())
        ps = [zc[z] / 20.0 for z in ZONE_NAMES]
        issue_row = {
            "issue": int(df.iloc[i]["issue"]),
            "draw_date": str(df.iloc[i]["draw_date"]),
            "target_issue": (
                int(df.iloc[i + 1]["issue"])
                if i + 1 < len(df)
                else int(df.iloc[i]["issue"]) + 1
            ),
            "target_numbers": (
                json.dumps(draws[i + 1], ensure_ascii=False)
                if i + 1 < len(df)
                else json.dumps([], ensure_ascii=False)
            ),
            "prev_numbers": json.dumps(draws[i - 1], ensure_ascii=False),
            "current_numbers": json.dumps(nums, ensure_ascii=False),
            "history_numbers": json.dumps(draws[: i + 1], ensure_ascii=False),
            "size_sequence": json.dumps(size_series[: i + 1], ensure_ascii=False),
            "odd_even_sequence": json.dumps(
                odd_even_series[: i + 1], ensure_ascii=False
            ),
            "issue_zone_entropy": _entropy(ps) / np.log(4.0),
            "issue_span_z50": _zscore(span, spans, z_window),
            "issue_sum_z50": _zscore(total, sums, z_window),
            "issue_consecutive_z50": _zscore(consecutive_pairs, consecutives, z_window),
            "size_label": size_series[i],
            "odd_even_label": odd_even_series[i],
        }
        rows.append(issue_row)
        spans.append(span)
        sums.append(total)
        consecutives.append(consecutive_pairs)
    return pd.DataFrame(rows)


def build_issue_features(
    df: pd.DataFrame,
    min_history: int = 20,
    include_latest_for_inference: bool = False,
) -> pd.DataFrame:
    feature_version = _feature_version()
    normalize_feature_version(feature_version)
    return _build_issue_features_v3(
        df,
        min_history=min_history,
        include_latest_for_inference=include_latest_for_inference,
    )


def build_latest_issue_features_for_inference(
    df: pd.DataFrame,
    min_history: int = 20,
) -> pd.DataFrame:
    return build_issue_features(
        df,
        min_history=min_history,
        include_latest_for_inference=True,
    )


def issue_feature_columns(df: pd.DataFrame) -> List[str]:
    skip = {
        "issue",
        "draw_date",
        "target_issue",
        "target_numbers",
        "prev_numbers",
        "current_numbers",
        "history_numbers",
        "size_sequence",
        "odd_even_sequence",
        "size_label",
        "odd_even_label",
    }
    return [c for c in df.columns if c not in skip]


def build_candidate_matrix(
    issue_row: pd.Series,
    feature_columns: Sequence[str],
    strict_features: bool | None = None,
) -> pd.DataFrame:
    feature_version = _feature_version()
    normalize_feature_version(feature_version)
    return _build_candidate_matrix_v3(issue_row, feature_columns, strict_features)


def build_stage1_candidate_matrix(
    issue_row: pd.Series,
    feature_columns: Sequence[str],
    strict_features: bool | None = None,
    pipeline_version: str = "cascade_v1",
) -> pd.DataFrame:
    validate_stage_feature_contract(pipeline_version, "stage1", feature_columns)
    full = _build_candidate_matrix_v3(
        issue_row,
        V3_CORE20_COLUMNS,
        strict_features=strict_features,
    )
    stage1 = full.reindex(columns=list(feature_columns)).copy()
    stage1.insert(0, "number", np.arange(1, 81, dtype=int))
    return stage1


def build_stage2_candidate_matrix(
    issue_row: pd.Series,
    stage1_df: pd.DataFrame,
    feature_columns: Sequence[str],
    strict_features: bool | None = None,
    pipeline_version: str = "cascade_v1",
) -> pd.DataFrame:
    validate_stage_feature_contract(pipeline_version, "stage2", feature_columns)
    if "number" not in stage1_df.columns:
        raise ValueError("stage1_df must include number column")
    keep_df = stage1_df
    if "stage1_keep_flag" in stage1_df.columns:
        keep_df = stage1_df[stage1_df["stage1_keep_flag"] > 0]
    keep_numbers = keep_df["number"].astype(int).tolist()
    full = _build_candidate_matrix_v3(
        issue_row,
        V3_CORE20_COLUMNS,
        strict_features=strict_features,
    )
    full.insert(0, "number", np.arange(1, 81, dtype=int))
    stage2 = full[full["number"].isin(keep_numbers)].copy()
    stage2 = stage2.reindex(columns=["number", *list(feature_columns)])
    for col in ["stage1_score", "stage1_rank", "stage1_keep_flag"]:
        if col in stage1_df.columns:
            stage2 = stage2.merge(
                stage1_df[["number", col]],
                on="number",
                how="left",
            )
    return stage2


def build_stage3_selector_inputs(
    issue_row: pd.Series,
    stage2_df: pd.DataFrame,
    top_k: int = 10,
    pipeline_version: str = "cascade_v1",
) -> pd.DataFrame:
    _ = issue_row
    if "number" not in stage2_df.columns:
        raise ValueError("stage2_df must include number column")
    work = stage2_df.copy()
    if "stage2_keep_flag" in work.columns:
        work = work[work["stage2_keep_flag"] > 0]
    if "stage2_score" in work.columns:
        work = work.sort_values("stage2_score", ascending=False)
    work = work.head(int(top_k)).copy()
    work["zone_id"] = ((work["number"].astype(int) - 1) // 20).astype(int)
    work["tail"] = (work["number"].astype(int) % 10).astype(int)
    if "stage2_score" not in work.columns:
        work["stage2_score"] = 0.0
    if "stage2_rank" not in work.columns:
        work["stage2_rank"] = np.arange(1, len(work) + 1, dtype=int)
    selector = work[["number", "zone_id", "tail", "stage2_score", "stage2_rank"]].copy()
    validate_stage_feature_contract(pipeline_version, "stage3", selector.columns)
    return selector


def _build_indicator_matrix(draws: Sequence[Sequence[int]], window: int) -> np.ndarray:
    data = np.zeros((min(window, len(draws)), 80), dtype=float)
    for i, draw in enumerate(draws[-window:]):
        for n in draw:
            data[i, n - 1] = 1.0
    return data


def _build_candidate_matrix_v3(
    issue_row: pd.Series,
    feature_columns: Sequence[str],
    strict_features: bool | None = None,
) -> pd.DataFrame:
    cfg = _load_feature_runtime_config()
    windows = resolve_effective_windows(0, cfg)
    decay = cfg["decay_half_lives"]
    alpha = float(cfg["smoothing_alpha"])
    tau = float(cfg["distance_kernel_tau"])

    base = issue_row.to_dict()
    history = [sorted(x) for x in json.loads(base.get("history_numbers", "[]"))]
    windows = resolve_effective_windows(len(history), cfg)
    last_draw = sorted(json.loads(base.get("current_numbers", "[]")))
    last_draw_set = set(last_draw)
    short_w = int(windows.get("freq_short", min(20, len(history))))
    long_w = int(windows.get("freq_long", min(200, len(history))))
    pmi_w = int(windows.get("pmi_window", min(200, len(history))))
    handoff_w = int(windows.get("handoff_window", min(200, len(history))))
    recent_k = min(20, len(history))

    m_short = _build_indicator_matrix(history, short_w)
    m_long = _build_indicator_matrix(history, long_w)
    m_pmi = _build_indicator_matrix(history, pmi_w)

    cnt_short = m_short.sum(axis=0) if len(m_short) else np.zeros(80)
    cnt_long = m_long.sum(axis=0) if len(m_long) else np.zeros(80)
    p_short = np.array(
        [_laplace_rate(v, max(1, len(m_short)), alpha) for v in cnt_short]
    )
    p_long = np.array([_laplace_rate(v, max(1, len(m_long)), alpha) for v in cnt_long])

    ewma_alpha = _half_life_alpha(float(decay.get("ewma", 50)))
    ewma = np.zeros(80, dtype=float)
    for draw in history:
        hit = np.zeros(80, dtype=float)
        for n in draw:
            hit[n - 1] = 1.0
        ewma = ewma_alpha * hit + (1 - ewma_alpha) * ewma

    # pairwise PPMI
    if len(m_pmi):
        co = m_pmi.T @ m_pmi
        n_rows = float(len(m_pmi))
        pi = m_pmi.mean(axis=0)
        pij = co / n_rows
    else:
        co = np.zeros((80, 80), dtype=float)
        pi = np.zeros(80, dtype=float)
        pij = np.zeros((80, 80), dtype=float)
    eps = 1e-12

    handoff_scores = np.zeros(80, dtype=float)
    for idx, num in enumerate(range(1, 81)):
        a_hits = 0.0
        b_after_a = 0.0
        b_hits = 0.0
        transitions = history[-(handoff_w + 1) :] if len(history) >= 2 else []
        for t in range(len(transitions) - 1):
            draw_t = set(transitions[t])
            draw_next = set(transitions[t + 1])
            neigh = {num - 1, num + 1}
            neigh = {x for x in neigh if 1 <= x <= 80}
            has_a = bool(draw_t & neigh)
            has_b = num in draw_next
            if has_a:
                a_hits += 1.0
                if has_b:
                    b_after_a += 1.0
            if has_b:
                b_hits += 1.0
        p_b_given_a = (b_after_a + alpha) / (a_hits + 2 * alpha) if a_hits > 0 else 0.5
        p_b = (b_hits + alpha) / (max(1.0, len(transitions) - 1) + 2 * alpha)
        handoff_scores[idx] = float(np.log((p_b_given_a + eps) / (p_b + eps)))

    neighbor_w = _exp_decay_weights(recent_k, float(decay.get("neighbor", 10)))
    recent_w = _exp_decay_weights(recent_k, float(decay.get("recent_hit", 5)))
    recent_draws = history[-recent_k:]

    size_series = [
        str(x)
        for x in json.loads(
            base.get("size_sequence", json.dumps([base.get("size_label", "小")]))
        )
    ]
    odd_even_series = [
        str(x)
        for x in json.loads(
            base.get(
                "odd_even_sequence", json.dumps([base.get("odd_even_label", "雙")])
            )
        )
    ]

    def _window_ratio(series: Sequence[str], label: str, window: int) -> float:
        local = list(series[-min(len(series), max(1, window)) :])
        return float(sum(1 for x in local if x == label) / max(1, len(local)))

    def _window_switches(series: Sequence[str], window: int) -> float:
        local = list(series[-min(len(series), max(1, window)) :])
        return float(sum(1 for a, b in zip(local, local[1:]) if a != b))

    prev_size = str(base.get("size_label", "小"))
    prev_oe = str(base.get("odd_even_label", "雙"))

    approx_draws_per_day = int(cfg.get("approx_draws_per_day", 78))

    rows = []
    for idx, num in enumerate(range(1, 81)):
        occurrences = [i for i, draw in enumerate(history) if num in draw]
        if occurrences:
            gap = len(history) - 1 - occurrences[-1]
            gaps = np.diff(occurrences)
            recent_gaps = gaps[-5:] if len(gaps) else np.array([], dtype=float)
            median_gap = (
                float(np.median(recent_gaps))
                if len(recent_gaps)
                else float(len(history))
            )
            mad_gap = (
                float(np.median(np.abs(recent_gaps - median_gap)))
                if len(recent_gaps)
                else 0.0
            )
        else:
            gap = len(history)
            mad_gap = 0.0

        pmi_values = []
        for n in last_draw:
            j = n - 1
            pmi = float(np.log((pij[idx, j] + eps) / ((pi[idx] * pi[j]) + eps)))
            pmi_values.append(max(0.0, pmi))

        neighbor_pm1 = 0.0
        neighbor_pm2 = 0.0
        recent_hit = 0.0
        for step, draw in enumerate(reversed(recent_draws)):
            dset = set(draw)
            if neighbor_w.size > step:
                w = float(neighbor_w[step])
                neighbor_pm1 += w * float(num - 1 in dset or num + 1 in dset)
                neighbor_pm2 += w * float(num - 2 in dset or num + 2 in dset)
            if recent_w.size > step:
                recent_hit += float(recent_w[step]) * float(num in dset)

        row = {
            "issue_zone_entropy": float(base.get("issue_zone_entropy", 0.0)),
            "issue_span_z50": float(base.get("issue_span_z50", 0.0)),
            "issue_sum_z50": float(base.get("issue_sum_z50", 0.0)),
            "issue_consecutive_z50": float(base.get("issue_consecutive_z50", 0.0)),
            "num_zone": float((num - 1) // 20),
            "cand_repeat_last_draw": float(num in last_draw_set),
            "cand_freq_smooth_20": float(p_short[idx]),
            "cand_freq_smooth_200": float(p_long[idx]),
            "cand_freq_trend_20_200": float(p_short[idx] - p_long[idx]),
            "cand_freq_ewma_hl50": float(ewma[idx]),
            "cand_gap_log1p": float(np.log1p(gap)),
            "cand_recency_ratio_200": float(gap * p_long[idx]),
            "cand_gap_mad_5": float(mad_gap),
            "cand_recent_hit_decay_hl5": float(recent_hit),
            "cand_neighbor_pm1_decay_hl10": float(neighbor_pm1),
            "cand_neighbor_pm2_decay_hl10": float(neighbor_pm2),
            "cand_distance_kernel_last_draw_tau2": float(
                sum(np.exp(-abs(num - n) / tau) for n in last_draw)
            ),
            "cand_pmi_last_draw_sum_200": float(sum(pmi_values)),
            "cand_pmi_last_draw_max_200": float(max(pmi_values) if pmi_values else 0.0),
            "cand_handoff_pm1_lift_200": float(handoff_scores[idx]),
            "ctx_prev_size_is_big": float(prev_size == "大"),
            "ctx_prev_odd_even_is_odd": float(prev_oe == "單"),
            "ctx_size_big_ratio_w20": _window_ratio(size_series, "大", 20),
            "ctx_odd_ratio_w20": _window_ratio(odd_even_series, "單", 20),
            "ctx_size_switches_w20": _window_switches(size_series, 20),
            "ctx_odd_even_switches_w20": _window_switches(odd_even_series, 20),
            "cand_hits_last_200": float(
                sum(1 for draw in history[-min(200, len(history)) :] if num in draw)
            ),
            "cand_hits_last_500": float(
                sum(1 for draw in history[-min(500, len(history)) :] if num in draw)
            ),
            "cand_hits_last_1000": float(
                sum(1 for draw in history[-min(1000, len(history)) :] if num in draw)
            ),
            "cand_total_hits_all_time": float(len(occurrences)),
            "cand_current_gap_all": float(gap),
            "cand_avg_gap_all": float(
                np.mean(gaps) if len(occurrences) >= 2 else len(history)
            ),
            "cand_max_gap_all": float(
                max(gaps) if len(occurrences) >= 2 else len(history)
            ),
            "cand_today_hits": float(
                sum(
                    1
                    for draw in history[-min(approx_draws_per_day, len(history)) :]
                    if num in draw
                )
            ),
            "cand_carryover_from_prev": float(
                num in last_draw_set and num in set(history[-2])
                if len(history) >= 2
                else 0.0
            ),
            "cand_pm1_neighbor_hits": float(
                sum(1 for n in last_draw_set if abs(num - n) == 1)
            ),
            "cand_pm2_neighbor_hits": float(
                sum(1 for n in last_draw_set if abs(num - n) == 2)
            ),
        }

        prev_distances = [abs(num - n) for n in last_draw] if last_draw else [80.0]
        row["cand_min_abs_distance_to_prev_draw"] = float(min(prev_distances))
        row["cand_mean_abs_distance_to_prev_draw"] = float(np.mean(prev_distances))
        row["cand_is_exact_in_prev_draw"] = float(num in last_draw_set)
        row["cand_has_prev_pm1"] = float(any(abs(num - n) <= 1 for n in last_draw_set))
        row["cand_has_prev_pm2"] = float(any(abs(num - n) <= 2 for n in last_draw_set))
        row["cand_has_prev_pm3"] = float(any(abs(num - n) <= 3 for n in last_draw_set))
        row["cand_count_prev_within_1"] = float(
            sum(1 for n in last_draw if abs(num - n) <= 1)
        )
        row["cand_count_prev_within_2"] = float(
            sum(1 for n in last_draw if abs(num - n) <= 2)
        )
        row["cand_count_prev_within_3"] = float(
            sum(1 for n in last_draw if abs(num - n) <= 3)
        )

        for k in [3, 5, 10]:
            recent = history[-min(k, len(history)) :]
            flat = [x for draw in recent for x in draw]
            distances = [abs(num - n) for n in flat] if flat else [80.0]
            row[f"cand_min_distance_to_recent_{k}"] = float(min(distances))
            if k == 10:
                weights = np.exp(-np.arange(len(recent), 0, -1) / max(1.0, tau))
                weighted = 0.0
                denom = float(np.sum(weights)) or 1.0
                for w_idx, draw in enumerate(recent):
                    nearest = min(abs(num - n) for n in draw) if draw else 80.0
                    weighted += float(weights[w_idx]) * float(nearest)
                row["cand_weighted_distance_decay_recent_10"] = float(weighted / denom)
                row["cand_count_recent_within_1_10"] = float(
                    sum(1 for n in flat if abs(num - n) <= 1)
                )
        rows.append(row)

    out = pd.DataFrame(rows)
    strict = (
        bool(int(os.getenv("STRICT_FEATURES", "0")))
        if strict_features is None
        else strict_features
    )
    missing = [col for col in feature_columns if col not in out.columns]
    if missing:
        if strict:
            raise ValueError(f"Missing feature columns: {missing}")
        LOGGER.warning("Missing feature columns, filling zeros: %s", missing)
        for col in missing:
            out[col] = 0.0
    return out[list(feature_columns)]


def precompute_issue_payloads(
    feature_df: pd.DataFrame,
    feature_columns: Sequence[str],
    strict_features: bool | None = None,
) -> dict[int, dict[str, object]]:
    payloads: dict[int, dict[str, object]] = {}
    for idx, row in feature_df.iterrows():
        cand = build_candidate_matrix(
            row,
            feature_columns,
            strict_features=strict_features,
        )
        payloads[int(idx)] = {
            "cand": cand,
            "target": set(json.loads(row["target_numbers"])),
            "regime": None,
            "issue_row": row,
        }
    return payloads


def compact_10_from_top20(top20: Sequence[int]) -> List[int]:
    buckets = {z: [] for z in ZONE_NAMES}
    for n in top20:
        buckets[zone_of(n)].append(n)
    compact = []
    while len(compact) < 10 and any(buckets.values()):
        for z in ZONE_NAMES:
            if buckets[z] and len(compact) < 10:
                compact.append(buckets[z].pop(0))
    return compact


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_processed() -> pd.DataFrame:
    canonical_parquet = DATA_PROCESSED_DIR / "bingo_draws_canonical.parquet"
    if canonical_parquet.exists():
        df = pd.read_parquet(canonical_parquet)
        if "numbers" in df.columns:
            return df

    canonical_csv = DATA_PROCESSED_DIR / "bingo_draws_canonical.csv"
    if canonical_csv.exists():
        df = pd.read_csv(canonical_csv)
        if "numbers" in df.columns:
            return df

    return pd.read_csv(DATA_PROCESSED_DIR / "bingo_draws.csv")


def _direction(current: float, baseline: float) -> str:
    if current > baseline:
        return "up"
    if current < baseline:
        return "down"
    return "flat"


def build_recent_report(recent_draws: Sequence[Sequence[int]]) -> dict:
    if not recent_draws:
        return {}

    draws = [sorted(list(d)) for d in recent_draws]
    arr = np.array(draws)
    odd_cnt_series = (arr % 2 == 1).sum(axis=1)
    even_cnt_series = 20 - odd_cnt_series
    small_cnt_series = (arr <= 40).sum(axis=1)
    big_cnt_series = 20 - small_cnt_series

    def _avg(series: np.ndarray, w: int) -> float:
        return float(series[-min(w, len(series)) :].mean())

    latest = draws[-1]
    zc = _zone_counts(latest)
    zone_values = list(zc.values())
    if max(zone_values) - min(zone_values) <= 1:
        zone_pattern = "均衡盤"
    elif max(zone_values) >= 9:
        zone_pattern = "單區爆發"
    elif (
        sorted(zone_values, reverse=True)[0] >= 7
        and sorted(zone_values, reverse=True)[1] >= 6
    ):
        zone_pattern = "雙區震盪"
    else:
        zone_pattern = "中段主導"

    freq_stats = {}
    for w in [3, 5, 10, 20]:
        freq = _recent_frequency(draws, min(w, len(draws)))
        freq_stats[f"recent_freq_{w}"] = {
            "max": int(max(freq.values())),
            "min": int(min(freq.values())),
            "mean": float(np.mean(list(freq.values()))),
        }

    freq20 = _recent_frequency(draws, min(20, len(draws)))
    sorted_freq = sorted(freq20.items(), key=lambda x: (-x[1], x[0]))
    hot_numbers = [n for n, _ in sorted_freq[:5]]
    cold_numbers = [
        n for n, _ in sorted(freq20.items(), key=lambda x: (x[1], x[0]))[:5]
    ]
    prev_set = set(draws[-2]) if len(draws) >= 2 else set()
    latest_set = set(latest)
    overlap_prev = float(len(latest_set & prev_set) / 20) if prev_set else 0.0
    overlap_prev_p1 = (
        float(len(latest_set & _shifted_set(prev_set, 1)) / 20) if prev_set else 0.0
    )
    overlap_prev_p2 = (
        float(len(latest_set & _shifted_set(prev_set, 2)) / 20) if prev_set else 0.0
    )

    return {
        "odd_even": {
            "odd_cnt": int(odd_cnt_series[-1]),
            "even_cnt": int(even_cnt_series[-1]),
            "odd_minus_even": int(odd_cnt_series[-1] - even_cnt_series[-1]),
            "odd_cnt_roll_mean": {str(w): _avg(odd_cnt_series, w) for w in [5, 10, 20]},
            "even_cnt_roll_mean": {
                str(w): _avg(even_cnt_series, w) for w in [5, 10, 20]
            },
            "odd_even_shift_direction": _direction(
                float(odd_cnt_series[-1] - even_cnt_series[-1]),
                _avg(odd_cnt_series - even_cnt_series, 5),
            ),
        },
        "big_small": {
            "small_cnt": int(small_cnt_series[-1]),
            "big_cnt": int(big_cnt_series[-1]),
            "big_minus_small": int(big_cnt_series[-1] - small_cnt_series[-1]),
            "small_cnt_roll_mean": {
                str(w): _avg(small_cnt_series, w) for w in [5, 10, 20]
            },
            "big_cnt_roll_mean": {str(w): _avg(big_cnt_series, w) for w in [5, 10, 20]},
            "big_small_shift_direction": _direction(
                float(big_cnt_series[-1] - small_cnt_series[-1]),
                _avg(big_cnt_series - small_cnt_series, 5),
            ),
        },
        "zone": {
            "zone_A_cnt": zc["A"],
            "zone_B_cnt": zc["B"],
            "zone_C_cnt": zc["C"],
            "zone_D_cnt": zc["D"],
            "zone_roll_mean": {
                str(w): {
                    f"zone_{z}": float(
                        np.mean(
                            [_zone_counts(d)[z] for d in draws[-min(w, len(draws)) :]]
                        )
                    )
                    for z in ZONE_NAMES
                }
                for w in [5, 10]
            },
            "board_regime": zone_pattern,
        },
        "recent_frequency": {
            **freq_stats,
            "hot_numbers": hot_numbers,
            "cold_numbers": cold_numbers,
            "overlap_with_prev_draw": overlap_prev,
            "overlap_with_prev_plus1": overlap_prev_p1,
            "overlap_with_prev_plus2": overlap_prev_p2,
        },
    }


def apply_local_peak_correction(
    score_table: pd.DataFrame,
    cfg: dict | None = None,
    input_score_column: str = "score_after_analysis_rerank",
    output_score_column: str = "score_after_local_peak",
) -> tuple[pd.DataFrame, dict]:
    local_cfg = cfg or {}
    enabled = bool(local_cfg.get("enabled", False))
    alpha_pm1 = float(local_cfg.get("alpha_pm1", 0.20))
    alpha_pm2 = float(local_cfg.get("alpha_pm2", 0.08))

    out = score_table.copy()
    if input_score_column not in out.columns:
        raise ValueError(f"missing input_score_column={input_score_column}")

    out["raw_score"] = out[input_score_column].astype(float)
    score_by_number = {
        int(rec["number"]): float(rec["raw_score"])
        for rec in out[["number", "raw_score"]].to_dict(orient="records")
    }

    local_scores: list[float] = []
    for num in out["number"].astype(int).tolist():
        pm1 = score_by_number.get(num - 1, 0.0) + score_by_number.get(num + 1, 0.0)
        pm2 = score_by_number.get(num - 2, 0.0) + score_by_number.get(num + 2, 0.0)
        local_scores.append(
            float(score_by_number.get(num, 0.0) + alpha_pm1 * pm1 + alpha_pm2 * pm2)
        )
    out["local_peak_score"] = pd.Series(local_scores, index=out.index).astype(float)

    if enabled:
        out[output_score_column] = out["local_peak_score"].astype(float)
    else:
        out[output_score_column] = out["raw_score"].astype(float)

    summary = {
        "enabled": enabled,
        "alpha_pm1": alpha_pm1,
        "alpha_pm2": alpha_pm2,
        "input_score_column": input_score_column,
        "output_score_column": output_score_column,
        "top5_preview": out.sort_values(output_score_column, ascending=False)
        .head(5)[["number", "raw_score", "local_peak_score", output_score_column]]
        .to_dict(orient="records"),
    }
    return out, summary


def build_group_dedup_priority(
    score_table: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.Series:
    dedup_cfg = cfg or {}
    final_weight = float(dedup_cfg.get("final_score_weight", 0.50))
    local_peak_weight = float(dedup_cfg.get("local_peak_weight", 0.25))
    history_weight = float(dedup_cfg.get("history_prior_weight", 0.15))
    gap_penalty_weight = float(dedup_cfg.get("gap_penalty_weight", 0.10))
    local_col = (
        "local_peak_score"
        if "local_peak_score" in score_table.columns
        else "score_after_local_peak"
    )
    return (
        final_weight * score_table["final_score"].astype(float)
        + local_peak_weight * score_table.get(local_col, 0.0).astype(float)
        + history_weight * score_table.get("history_prior_score", 0.0).astype(float)
        - gap_penalty_weight
        * score_table.get("cand_current_gap_all", 0.0).astype(float)
    )


def apply_topk_group_dedup(
    score_table: pd.DataFrame,
    cfg: dict | None = None,
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    dedup_cfg = cfg or {}
    enabled = bool(dedup_cfg.get("enabled", False))
    group_distance = max(0, int(dedup_cfg.get("group_distance", 1)))
    apply_to_top3_only = bool(dedup_cfg.get("apply_to_top3_only", True))
    candidate_pool_for_grouping = max(
        int(top_k), int(dedup_cfg.get("candidate_pool_for_grouping", 20))
    )

    ranked = score_table.sort_values("final_score", ascending=False).reset_index(
        drop=True
    )
    top3_before = ranked["number"].head(top_k).astype(int).tolist()
    if (not enabled) or group_distance <= 0:
        summary = {
            "enabled": enabled,
            "group_distance": group_distance,
            "apply_to_top3_only": apply_to_top3_only,
            "candidate_pool_for_grouping": candidate_pool_for_grouping,
            "grouped_candidates_preview": [],
            "top3_before_group_dedup": top3_before,
            "top3_after_group_dedup": top3_before,
            "dedup_applied_scope": "top3_only" if apply_to_top3_only else "ranking",
        }
        return ranked, summary

    work = ranked.copy()
    work["group_dedup_priority"] = build_group_dedup_priority(work, cfg=dedup_cfg)

    pool = work.head(candidate_pool_for_grouping).copy()
    groups: list[list[dict]] = []
    for rec in pool.to_dict(orient="records"):
        num = int(rec["number"])
        placed = False
        for grp in groups:
            if any(abs(num - int(x["number"])) <= group_distance for x in grp):
                grp.append(rec)
                placed = True
                break
        if not placed:
            groups.append([rec])

    representatives: list[dict] = []
    grouped_preview: list[dict] = []
    for grp in groups:
        grp_sorted = sorted(
            grp,
            key=lambda x: float(x.get("group_dedup_priority", 0.0)),
            reverse=True,
        )
        rep = grp_sorted[0]
        representatives.append(rep)
        grouped_preview.append(
            {
                "members": [int(x["number"]) for x in grp_sorted],
                "representative": int(rep["number"]),
                "representative_reason": {
                    "group_dedup_priority": float(rep.get("group_dedup_priority", 0.0)),
                    "final_score": float(rep.get("final_score", 0.0)),
                    "local_peak_score": float(rep.get("local_peak_score", 0.0)),
                    "history_prior_score": float(rep.get("history_prior_score", 0.0)),
                    "cand_current_gap_all": float(rep.get("cand_current_gap_all", 0.0)),
                },
            }
        )

    rep_numbers = [int(x["number"]) for x in representatives]
    if apply_to_top3_only:
        top3_after: list[int] = []
        for num in rep_numbers:
            if len(top3_after) >= top_k:
                break
            top3_after.append(int(num))
        for num in top3_before:
            if len(top3_after) >= top_k:
                break
            if num not in top3_after:
                top3_after.append(num)
        summary = {
            "enabled": enabled,
            "group_distance": group_distance,
            "apply_to_top3_only": apply_to_top3_only,
            "candidate_pool_for_grouping": candidate_pool_for_grouping,
            "grouped_candidates_preview": grouped_preview[:8],
            "top3_before_group_dedup": top3_before,
            "top3_after_group_dedup": top3_after,
            "dedup_applied_scope": "top3_only",
        }
        return ranked, summary

    dedup_order = rep_numbers[:]
    for num in ranked["number"].astype(int).tolist():
        if num not in dedup_order:
            dedup_order.append(num)
    order_map = {n: i for i, n in enumerate(dedup_order)}
    reranked = (
        ranked.assign(_dedup_order=ranked["number"].astype(int).map(order_map))
        .sort_values("_dedup_order", ascending=True)
        .drop(columns=["_dedup_order"])
        .reset_index(drop=True)
    )
    top3_after = reranked["number"].head(top_k).astype(int).tolist()
    summary = {
        "enabled": enabled,
        "group_distance": group_distance,
        "apply_to_top3_only": apply_to_top3_only,
        "candidate_pool_for_grouping": candidate_pool_for_grouping,
        "grouped_candidates_preview": grouped_preview[:8],
        "top3_before_group_dedup": top3_before,
        "top3_after_group_dedup": top3_after,
        "dedup_applied_scope": "ranking",
    }
    return reranked, summary
