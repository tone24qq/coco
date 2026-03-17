from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.artifacts import CascadeArtifacts  # noqa: E402
from src.artifacts import save_cascade_artifacts  # noqa: E402
from src.pipeline import CascadePipeline  # noqa: E402
from src.strategy import (  # noqa: E402
    StrategyConfig,
    apply_strategy,
    default_experiments,
    derive_regime,
    issue_metrics,
    strategy_to_dict,
)
from src.utils import (  # noqa: E402
    CASCADE_V1_STAGE1_COLUMNS,
    CASCADE_V1_STAGE2_COLUMNS,
    CASCADE_V1_STAGE3_COLUMNS,
    CONFIG_DIR,
    FEATURE_STORE_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    V3_CORE20_COLUMNS,
    build_stage1_candidate_matrix,
    build_stage2_candidate_matrix,
    load_yaml,
    normalize_pipeline_version,
    precompute_issue_payloads,
    save_json,
    validate_feature_columns_contract,
)

METRIC_KEYS = [
    "top20_hit_rate",
    "top5_hit_rate",
    "top10_hit_rate",
    "top3_hit_rate",
    "top3_at_least_one_hit_rate",
    "ndcg_at_10",
]

PREFERRED_STRATEGY_VERSION = "v3_rerank_k30_p300"


class PhaseMonitor:
    def __init__(
        self,
        heartbeat_seconds: int = 30,
        stalled_seconds: int = 600,
        watchdog_seconds: int = 900,
    ) -> None:
        self.heartbeat_seconds = heartbeat_seconds
        self.stalled_seconds = stalled_seconds
        self.watchdog_seconds = watchdog_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase_name = ""
        self._phase_start = 0.0
        self._last_heartbeat = 0.0
        self._last_progress = 0.0
        self._last_warn = 0.0
        self._last_dump = 0.0
        self._fold_index: int | None = None
        self._stage_index: int | None = None
        self._shape: tuple[int, int] | None = None

    def update_progress(
        self,
        *,
        fold_index: int | None = None,
        stage_index: int | None = None,
        shape: tuple[int, int] | None = None,
    ) -> None:
        with self._lock:
            self._last_progress = time.monotonic()
            if fold_index is not None:
                self._fold_index = int(fold_index)
            if stage_index is not None:
                self._stage_index = int(stage_index)
            if shape is not None:
                self._shape = shape

    @contextmanager
    def phase(self, name: str):
        now = time.monotonic()
        with self._lock:
            self._phase_name = name
            self._phase_start = now
            self._last_progress = now
            self._last_heartbeat = 0.0
            self._last_warn = 0.0
            self._last_dump = 0.0
            self._fold_index = None
            self._stage_index = None
            self._shape = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        print(f"[phase:start] {name}")
        try:
            yield
        finally:
            elapsed = time.monotonic() - self._phase_start
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1)
            print(f"[phase:end] {name} elapsed={elapsed:.2f}s")

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(1):
            now = time.monotonic()
            with self._lock:
                if now - self._last_heartbeat >= self.heartbeat_seconds:
                    print(
                        f"[phase:heartbeat] {self._phase_name} "
                        f"elapsed={now - self._phase_start:.2f}s"
                    )
                    self._last_heartbeat = now
                stalled_for = now - self._last_progress
                if stalled_for >= self.stalled_seconds and (
                    self._last_warn == 0.0
                    or now - self._last_warn >= self.stalled_seconds
                ):
                    print(
                        "[phase:stalled] "
                        f"phase={self._phase_name}, "
                        f"fold_index={self._fold_index}, "
                        f"stage_index={self._stage_index}, "
                        f"shape={self._shape}, "
                        f"stalled_for={stalled_for:.2f}s"
                    )
                    self._last_warn = now
                if (
                    self.watchdog_seconds > 0
                    and stalled_for >= self.watchdog_seconds
                    and (
                        self._last_dump == 0.0
                        or now - self._last_dump >= self.watchdog_seconds
                    )
                ):
                    print(
                        f"[watchdog] phase={self._phase_name} stalled_for={stalled_for:.2f}s; dumping traceback"
                    )
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    print(
                        "[watchdog:stack]" + "".join(traceback.format_stack(limit=12))
                    )
                    self._last_dump = now


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost / cascade pipeline")
    parser.add_argument("--debug", action="store_true", help="Run with small slices")
    parser.add_argument("--max-issues", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--thread-count", type=int, default=None)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    return parser.parse_args()


def _shape_of(x: object) -> tuple[int, int] | None:
    if hasattr(x, "shape"):
        shape = getattr(x, "shape")
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return None


def _build_catboost_params(cfg: dict, args: argparse.Namespace) -> dict:
    params = dict(cfg.get("catboost_params", {}))
    params.setdefault("loss_function", "Logloss")
    params.setdefault("verbose", False)
    params.setdefault("random_seed", 42)
    params.setdefault("thread_count", min(8, os.cpu_count() or 4))
    params.setdefault("logging_level", "Info")
    params.setdefault("metric_period", 50)
    params.setdefault("train_dir", str(MODELS_DIR / "catboost_train_logs"))
    params.setdefault("save_snapshot", True)
    params.setdefault("snapshot_file", str(MODELS_DIR / "catboost_snapshot.bin"))
    if args.thread_count is not None:
        params["thread_count"] = int(args.thread_count)
    if args.iterations is not None:
        params["iterations"] = int(args.iterations)
    if args.debug and args.iterations is None:
        params["iterations"] = 100
    print(
        "[catboost] "
        f"thread_count={params.get('thread_count')}, "
        f"logging_level={params.get('logging_level')}, "
        f"metric_period={params.get('metric_period')}, "
        f"train_dir={params.get('train_dir')}, "
        f"save_snapshot={params.get('save_snapshot')}, "
        f"snapshot_file={params.get('snapshot_file')}"
    )
    return params


def _fit_cascade_pipeline(
    feat_df: pd.DataFrame,
    indices: list[int],
    params: dict,
    stage1_keep: int,
    stage2_keep: int,
) -> CascadePipeline:
    local_df = feat_df.iloc[indices].reset_index(drop=True)
    pipeline, _ = CascadePipeline.train(
        local_df,
        stage1_keep=stage1_keep,
        stage2_keep=stage2_keep,
        catboost_params=params,
    )
    return pipeline


def _expand_rows(
    issue_payloads: dict[int, dict[str, object]],
    indices: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    x_blocks, y_blocks = [], []
    for idx in indices:
        payload = issue_payloads[int(idx)]
        cand = payload["cand"]
        target = payload["target"]
        labels = pd.Series([1 if n in target else 0 for n in range(1, 81)])
        x_blocks.append(cand)
        y_blocks.append(labels)
    return pd.concat(x_blocks, ignore_index=True), pd.concat(
        y_blocks, ignore_index=True
    )


def _expand_rows_with_soft_labels(
    issue_payloads: dict[int, dict[str, object]],
    indices: list[int],
    pm1_weight: float,
    pm2_weight: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x_blocks, soft_blocks, pm1_blocks = [], [], []
    for idx in indices:
        payload = issue_payloads[int(idx)]
        cand = payload["cand"]
        target = set(int(x) for x in payload["target"])
        soft = []
        pm1 = []
        for n in range(1, 81):
            if n in target:
                soft.append(1.0)
                pm1.append(1)
            elif any(abs(n - a) == 1 for a in target):
                soft.append(float(pm1_weight))
                pm1.append(1)
            elif any(abs(n - a) == 2 for a in target):
                soft.append(float(pm2_weight))
                pm1.append(0)
            else:
                soft.append(0.0)
                pm1.append(0)
        x_blocks.append(cand)
        soft_blocks.append(pd.Series(soft))
        pm1_blocks.append(pd.Series(pm1))
    return (
        pd.concat(x_blocks, ignore_index=True),
        pd.concat(soft_blocks, ignore_index=True),
        pd.concat(pm1_blocks, ignore_index=True),
    )


def _aggregate(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS}
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in METRIC_KEYS}


def _overfit_audit(
    train_fold: list[dict], test_fold: list[dict], regime_rows: list[dict], th: dict
) -> dict:
    train_top3 = np.array([x["top3_hit_rate"] for x in train_fold], dtype=float)
    test_top3 = np.array([x["top3_hit_rate"] for x in test_fold], dtype=float)
    regime_df = pd.DataFrame(regime_rows)
    gap = float(train_top3.mean() - test_top3.mean()) if len(train_top3) else 0.0
    fold_disp = float(test_top3.std(ddof=0)) if len(test_top3) else 0.0
    regime_disp = (
        float(regime_df["top3_hit_rate"].std(ddof=0)) if not regime_df.empty else 0.0
    )
    overfit = bool(
        gap > float(th.get("train_vs_oos_gap", 0.03))
        or fold_disp > float(th.get("fold_dispersion", 0.05))
        or regime_disp > float(th.get("regime_dispersion", 0.06))
    )
    return {
        "train_vs_oos_gap": gap,
        "fold_dispersion": fold_disp,
        "regime_dispersion": regime_disp,
        "is_overfit": overfit,
    }


def _load_experiments() -> list[StrategyConfig]:
    exp_cfg_path = CONFIG_DIR / "experiments.yaml"
    if not exp_cfg_path.exists():
        return default_experiments()
    payload = load_yaml(exp_cfg_path)
    experiments = []
    for row in payload.get("experiments", []):
        experiments.append(StrategyConfig(**row))
    return experiments or default_experiments()


def _evaluate_strategies(
    feat_df: pd.DataFrame,
    issue_payloads: dict[int, dict[str, object]],
    params: dict,
    splits: int,
    experiments: list[StrategyConfig],
    overfit_th: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    tss = TimeSeriesSplit(n_splits=splits)
    rows = []
    best = None
    baseline = None
    for exp in experiments:
        print(f"[版本開始] {exp.version_id}")
        fold_train, fold_test, regime_rows = [], [], []
        for fold_id, (tr_idx, te_idx) in enumerate(tss.split(feat_df), start=1):
            x_train, y_train = _expand_rows(issue_payloads, list(tr_idx))
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, verbose=False)
            cascade_pipeline = None
            if exp.stage_type == "cascade":
                cascade_pipeline = _fit_cascade_pipeline(
                    feat_df,
                    list(tr_idx),
                    params=params,
                    stage1_keep=exp.stage1_keep,
                    stage2_keep=exp.stage2_keep,
                )

            def _score_issue(row_idx: int) -> tuple[dict, str]:
                payload = issue_payloads[int(row_idx)]
                cand = payload["cand"]
                base_scores = model.predict_proba(cand)[:, 1]
                regime = payload["regime"]
                if regime is None:
                    regime = derive_regime(feat_df.iloc[row_idx])
                    payload["regime"] = regime
                if exp.stage_type == "cascade":
                    if cascade_pipeline is None:
                        raise ValueError("cascade pipeline not available")
                    cascade = cascade_pipeline.predict_issue(payload["issue_row"])
                    final_scores = cascade["final_scores"]
                else:
                    final_scores = apply_strategy(base_scores, cand, exp, regime)
                metric = issue_metrics(final_scores, payload["target"])
                return metric, regime

            train_eval_rows = []
            for row_idx in tr_idx[-min(50, len(tr_idx)) :]:
                m, _ = _score_issue(int(row_idx))
                train_eval_rows.append(m)
            test_eval_rows = []
            for row_idx in te_idx:
                m, regime = _score_issue(int(row_idx))
                test_eval_rows.append(m)
                regime_rows.append({"fold": fold_id, "regime": regime, **m})

            tr_agg = _aggregate(train_eval_rows)
            te_agg = _aggregate(test_eval_rows)
            fold_train.append({"fold": fold_id, **tr_agg})
            fold_test.append({"fold": fold_id, **te_agg})
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top20命中率={te_agg['top20_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top10命中率={te_agg['top10_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top3命中率={te_agg['top3_hit_rate']:.4f}"
            )
            print(
                f"[Fold {fold_id}/{splits}] {exp.version_id} top3至少中1顆率={te_agg['top3_at_least_one_hit_rate']:.4f}"
            )

        overall = _aggregate(fold_test)
        audit = _overfit_audit(fold_train, fold_test, regime_rows, overfit_th)
        if baseline is None:
            baseline = overall
        better = bool(
            overall["top3_at_least_one_hit_rate"]
            >= baseline["top3_at_least_one_hit_rate"]
            and overall["top3_hit_rate"] >= baseline["top3_hit_rate"]
        )
        keep = bool(better and not audit["is_overfit"])
        rec = {
            **strategy_to_dict(exp),
            **overall,
            **audit,
            "is_better_than_baseline": better,
            "keep_recommendation": keep,
        }
        rows.append(rec)
        if best is None or (
            rec["keep_recommendation"],
            rec["top3_at_least_one_hit_rate"],
            rec["top3_hit_rate"],
        ) > (
            best["keep_recommendation"],
            best["top3_at_least_one_hit_rate"],
            best["top3_hit_rate"],
        ):
            best = rec

    if best is None:
        raise ValueError("no strategy evaluated")
    if not bool(best["keep_recommendation"]):
        best = rows[0]
    return pd.DataFrame(rows), best, rows[0]


def _select_formal_strategy(registry_df: pd.DataFrame) -> dict:
    preferred_df = registry_df[registry_df["version_id"] == PREFERRED_STRATEGY_VERSION]
    if not preferred_df.empty:
        return preferred_df.iloc[0].to_dict()
    return (
        registry_df.sort_values(
            ["keep_recommendation", "top3_at_least_one_hit_rate", "top3_hit_rate"],
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )


def build_training_frame(
    *,
    cfg: dict,
    args: argparse.Namespace,
    monitor: PhaseMonitor,
) -> tuple[pd.DataFrame, list[str], str]:
    with monitor.phase("load canonical / feature store"):
        feature_df = pd.read_csv(FEATURE_STORE_DIR / "issue_features.csv")
        monitor.update_progress(shape=_shape_of(feature_df))
        feature_columns = json.loads(
            (MODELS_DIR / "feature_columns.json").read_text(encoding="utf-8")
        )
        feature_version = str(cfg.get("feature_version", "v3_core20"))
        if feature_version != "v3_core20":
            raise ValueError("only v3_core20 is supported")
        validate_feature_columns_contract(feature_columns, feature_version)

    with monitor.phase("dataset filter"):
        max_draws = int(cfg.get("max_draws_for_training", len(feature_df)))
        if args.max_issues is not None:
            max_draws = min(max_draws, int(args.max_issues))
        if args.debug and args.max_issues is None:
            max_draws = min(max_draws, 300)
        feature_df = feature_df.tail(max_draws).reset_index(drop=True)
        monitor.update_progress(shape=_shape_of(feature_df))
        print(
            f"[dataset] rows={len(feature_df)}, cols={feature_df.shape[1]}, max_draws_for_training={max_draws}"
        )
    return feature_df, feature_columns, feature_version


def train_stage1_model(
    feature_df: pd.DataFrame,
    params: dict,
    monitor: PhaseMonitor,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    with monitor.phase("cascade stage1 train"):
        stage1_blocks: list[pd.DataFrame] = []
        stage1_labels: list[pd.Series] = []
        for idx, issue_row in enumerate(feature_df.itertuples(index=False), start=1):
            issue_series = pd.Series(issue_row._asdict())
            cand = build_stage1_candidate_matrix(
                issue_series, CASCADE_V1_STAGE1_COLUMNS
            )
            target = set(json.loads(str(issue_series["target_numbers"])))
            y = cand["number"].astype(int).isin(target).astype(int)
            stage1_blocks.append(cand[CASCADE_V1_STAGE1_COLUMNS])
            stage1_labels.append(y)
            if idx % 25 == 0 or idx == len(feature_df):
                monitor.update_progress(stage_index=1, shape=_shape_of(cand))
                print(f"[progress] stage1 expand issue={idx}/{len(feature_df)}")

        stage1_x = pd.concat(stage1_blocks, ignore_index=True)
        stage1_y = pd.concat(stage1_labels, ignore_index=True)
        monitor.update_progress(stage_index=1, shape=_shape_of(stage1_x))
        print(f"[stage1] train shape={stage1_x.shape}")
        model = CatBoostClassifier(**params)
        model.fit(stage1_x, stage1_y, verbose=False)
        monitor.update_progress(stage_index=1, shape=_shape_of(stage1_x))
    return model, stage1_x


def train_stage2_model(
    feature_df: pd.DataFrame,
    stage1_model: CatBoostClassifier,
    params: dict,
    stage1_keep: int,
    monitor: PhaseMonitor,
) -> tuple[CatBoostClassifier, pd.DataFrame]:
    with monitor.phase("cascade stage2 train"):
        stage2_blocks: list[pd.DataFrame] = []
        stage2_labels: list[pd.Series] = []
        stage1_gate = CascadePipeline.from_artifacts(
            CascadeArtifacts(
                pipeline_version="cascade_v1",
                stage1_model=stage1_model,
                stage2_model=CatBoostClassifier(),
                stage1_feature_columns=list(CASCADE_V1_STAGE1_COLUMNS),
                stage2_feature_columns=list(CASCADE_V1_STAGE2_COLUMNS),
                stage3_input_schema=list(CASCADE_V1_STAGE3_COLUMNS),
                stage1_keep=int(stage1_keep),
                stage2_keep=10,
            )
        ).stage1
        for idx, issue_row in enumerate(feature_df.itertuples(index=False), start=1):
            issue_series = pd.Series(issue_row._asdict())
            stage1_df = stage1_gate.predict(issue_series)
            cand = build_stage2_candidate_matrix(
                issue_series, stage1_df, CASCADE_V1_STAGE2_COLUMNS
            )
            target = set(json.loads(str(issue_series["target_numbers"])))
            y = cand["number"].astype(int).isin(target).astype(int)
            stage2_blocks.append(cand[CASCADE_V1_STAGE2_COLUMNS])
            stage2_labels.append(y)
            if idx % 25 == 0 or idx == len(feature_df):
                monitor.update_progress(stage_index=2, shape=_shape_of(cand))
                print(f"[progress] stage2 expand issue={idx}/{len(feature_df)}")

        stage2_x = pd.concat(stage2_blocks, ignore_index=True)
        stage2_y = pd.concat(stage2_labels, ignore_index=True)
        monitor.update_progress(stage_index=2, shape=_shape_of(stage2_x))
        print(f"[stage2] train shape={stage2_x.shape}")
        model = CatBoostClassifier(**params)
        model.fit(stage2_x, stage2_y, verbose=False)
        monitor.update_progress(stage_index=2, shape=_shape_of(stage2_x))
    return model, stage2_x


def train_selector(monitor: PhaseMonitor, feature_df: pd.DataFrame) -> list[str]:
    with monitor.phase("selector/rerank train"):
        monitor.update_progress(stage_index=3, shape=_shape_of(feature_df))
        print("[selector] rule-based selector uses stage3 schema; no model fit needed")
        return list(CASCADE_V1_STAGE3_COLUMNS)


def _train_cascade_mode(
    feature_df: pd.DataFrame,
    feature_version: str,
    params: dict,
    pipeline_cfg: dict,
    monitor: PhaseMonitor,
) -> None:
    stage1_keep = int(pipeline_cfg.get("stage1_keep", 30))
    stage2_keep = int(pipeline_cfg.get("stage2_keep", 10))
    print("[訓練開始] pipeline=cascade_v1 (stage-aware)")

    stage1_model, _ = train_stage1_model(feature_df, params, monitor)
    stage2_model, _ = train_stage2_model(
        feature_df, stage1_model, params, stage1_keep, monitor
    )
    selector_schema = train_selector(monitor, feature_df)

    artifacts = CascadeArtifacts(
        pipeline_version="cascade_v1",
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage1_feature_columns=list(CASCADE_V1_STAGE1_COLUMNS),
        stage2_feature_columns=list(CASCADE_V1_STAGE2_COLUMNS),
        stage3_input_schema=selector_schema,
        stage1_keep=stage1_keep,
        stage2_keep=stage2_keep,
    )
    with monitor.phase("save model / metadata"):
        artifact_dir_cfg = str(pipeline_cfg.get("artifact_dir", "models/cascade_v1"))
        artifact_dir = PROJECT_ROOT / artifact_dir_cfg
        artifact_meta = save_cascade_artifacts(
            artifact_dir,
            artifacts,
            feature_version=feature_version,
            train_issue_start=int(feature_df["issue"].min()),
            train_issue_end=int(feature_df["target_issue"].max()),
        )

        selected_strategy = {
            "version_id": "cascade_v1_flow",
            "stage_type": "cascade",
            "pipeline_version": "cascade_v1",
            "model_artifact_dir": artifact_dir_cfg,
            "stage1_keep": stage1_keep,
            "stage2_keep": stage2_keep,
            "candidate_pool": stage1_keep,
            "prior_window": 300,
            "rerank_weight": 0.0,
            "penalty_weight": 0.0,
            "trend_weight": 0.0,
            "regime_aware": True,
        }
        save_json(
            MODELS_DIR / "strategy_config.json",
            {
                "selected_strategy": selected_strategy,
                "fallback_strategy": selected_strategy,
            },
        )

        legacy_metadata = {}
        metadata_path = MODELS_DIR / "metadata.json"
        if metadata_path.exists():
            legacy_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = {
            **legacy_metadata,
            "model_type": "catboost_cascade",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_rows": int(len(feature_df)),
            "feature_version": feature_version,
            "selected_strategy": selected_strategy,
            "pipeline_artifacts": {
                "cascade_v1": {
                    "artifact_dir": "models/cascade_v1",
                    **artifact_meta,
                }
            },
        }
        save_json(metadata_path, metadata)
        monitor.update_progress(shape=_shape_of(feature_df))
        print("[完成] cascade artifacts saved to models/cascade_v1")


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = _parse_args()
    cfg = load_yaml(CONFIG_DIR / "train.yaml")
    if bool(cfg.get("ranking_experiment", {}).get("enabled", False)):
        raise NotImplementedError(
            "ranking_experiment.enabled=true is not formally supported yet"
        )
    pipeline_cfg = cfg.get("pipeline", {})
    monitor = PhaseMonitor(watchdog_seconds=int(args.watchdog_seconds))
    pipeline_version = normalize_pipeline_version(
        pipeline_cfg.get("version", "baseline_flat_score")
    )
    print(
        f"[pipeline] resolved from configs/train.yaml: version={pipeline_version}, "
        f"stage1_keep={int(pipeline_cfg.get('stage1_keep', 30))}, "
        f"stage2_keep={int(pipeline_cfg.get('stage2_keep', 10))}"
    )
    os.environ["STRICT_FEATURES"] = "1"
    feature_df, feature_columns, feature_version = build_training_frame(
        cfg=cfg,
        args=args,
        monitor=monitor,
    )
    max_draws = len(feature_df)

    if len(feature_df) < 3000 and not args.debug:
        raise ValueError("訓練資料不足 3000 期，請先更新資料。")

    params = _build_catboost_params(cfg, args)

    if pipeline_version.startswith("cascade"):
        with monitor.phase("label/candidate expand"):
            monitor.update_progress(shape=_shape_of(feature_df), stage_index=0)
            print("[info] cascade mode expands candidates inside stage training")
        with monitor.phase("fold build"):
            monitor.update_progress(shape=_shape_of(feature_df), fold_index=0)
            print("[info] cascade mode uses full frame training (no CV folds)")
        _train_cascade_mode(
            feature_df=feature_df,
            feature_version=feature_version,
            params=params,
            pipeline_cfg=pipeline_cfg,
            monitor=monitor,
        )
        return

    print("[訓練開始] 模型：CatBoost Binary")
    print(f"[訓練設定] max_draws_for_training={max_draws}")
    print(
        f"[資料摘要] 訓練期數：{len(feature_df)}，特徵數：{len(feature_columns)}，模型類型：catboost"
    )

    all_experiments = _load_experiments()
    fast_version_ids = {
        "v0_binary_baseline",
        "v3_rerank_k30_p300",
        "v4_two_stage_20_10_3",
        "cascade_v1_flow",
    }
    fast_experiments = [
        exp for exp in all_experiments if exp.version_id in fast_version_ids
    ]
    with monitor.phase("label/candidate expand"):
        issue_payloads = precompute_issue_payloads(
            feature_df,
            feature_columns,
            strict_features=True,
        )
        monitor.update_progress(shape=(len(issue_payloads), len(feature_columns)))
    with monitor.phase("fold build"):
        monitor.update_progress(
            shape=(len(feature_df), len(feature_columns)),
            fold_index=int(cfg.get("backtest_splits", 5)),
        )
        print(
            f"[folds] research_splits={int(cfg.get('research_backtest_splits', 3))}, "
            f"final_splits={int(cfg.get('backtest_splits', 5))}"
        )

    print("[研究流程] 快速階段：3個版本、3 folds、較低 iterations")
    fast_params = dict(params)
    fast_params["iterations"] = int(cfg.get("research_iterations", 140))
    fast_registry_df, fast_best, _ = _evaluate_strategies(
        feature_df,
        issue_payloads,
        params=fast_params,
        splits=int(cfg.get("research_backtest_splits", 3)),
        experiments=fast_experiments,
        overfit_th=cfg.get("overfit_thresholds", {}),
    )

    candidates = [
        row
        for row in fast_registry_df.sort_values(
            ["keep_recommendation", "top3_at_least_one_hit_rate", "top3_hit_rate"],
            ascending=False,
        ).to_dict(orient="records")
        if row["version_id"] in fast_version_ids
    ]
    selected_final_ids = [
        x["version_id"] for x in candidates[: int(cfg.get("final_stage_versions", 2))]
    ]
    if not selected_final_ids:
        selected_final_ids = [fast_best["version_id"]]
    if "v0_binary_baseline" not in selected_final_ids:
        selected_final_ids = ["v0_binary_baseline", *selected_final_ids]
    final_experiments = [
        exp for exp in all_experiments if exp.version_id in set(selected_final_ids)
    ]

    print(
        f"[研究流程] 正式階段：版本={selected_final_ids}、{int(cfg.get('backtest_splits', 5))} folds"
    )
    registry_df, best, baseline = _evaluate_strategies(
        feature_df,
        issue_payloads,
        params=params,
        splits=int(cfg.get("backtest_splits", 5)),
        experiments=final_experiments,
        overfit_th=cfg.get("overfit_thresholds", {}),
    )
    selected_strategy = _select_formal_strategy(registry_df)

    x_train, y_train = _expand_rows(issue_payloads, list(range(len(feature_df))))
    final_model = CatBoostClassifier(**params)
    final_model.fit(x_train, y_train, verbose=False)
    final_model.save_model(str(MODELS_DIR / "catboost_top20.cbm"))

    x_soft = pd.DataFrame()
    x_pm1 = pd.DataFrame()
    soft_cfg = cfg.get("soft_label_training", {})
    if bool(soft_cfg.get("enabled", False)):
        x_soft, y_soft, _ = _expand_rows_with_soft_labels(
            issue_payloads,
            list(range(len(feature_df))),
            pm1_weight=float(soft_cfg.get("pm1_weight", 0.35)),
            pm2_weight=float(soft_cfg.get("pm2_weight", 0.15)),
        )
        soft_params = dict(params)
        soft_params["loss_function"] = "RMSE"
        soft_params["eval_metric"] = "RMSE"
        soft_model = CatBoostRegressor(**soft_params)
        soft_model.fit(x_soft, y_soft, verbose=False)
        soft_model.save_model(str(MODELS_DIR / "catboost_soft_label.cbm"))

    proximity_cfg = cfg.get("proximity_model", {})
    if bool(proximity_cfg.get("enabled", False)):
        x_pm1, _, y_pm1 = _expand_rows_with_soft_labels(
            issue_payloads,
            list(range(len(feature_df))),
            pm1_weight=float(soft_cfg.get("pm1_weight", 0.35)),
            pm2_weight=float(soft_cfg.get("pm2_weight", 0.15)),
        )
        pm1_model = CatBoostClassifier(**params)
        pm1_model.fit(x_pm1, y_pm1, verbose=False)
        pm1_model.save_model(str(MODELS_DIR / "catboost_pm1_proximity.cbm"))

    importances = final_model.get_feature_importance()
    fi_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": [float(x) for x in importances],
        }
    ).sort_values("importance", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry_df.to_csv(REPORTS_DIR / "experiment_registry.csv", index=False)
    fast_registry_df.to_csv(
        REPORTS_DIR / "experiment_registry_research.csv", index=False
    )
    fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_json(
        REPORTS_DIR / "feature_importance.json",
        {
            "features": fi_df.to_dict(orient="records"),
        },
    )

    print(
        "[整體結果] "
        f"top20_hit_rate={selected_strategy['top20_hit_rate']:.4f}, "
        f"top10_hit_rate={selected_strategy['top10_hit_rate']:.4f}, "
        f"top3_hit_rate={selected_strategy['top3_hit_rate']:.4f}, "
        "top3_at_least_one_hit_rate="
        f"{selected_strategy['top3_at_least_one_hit_rate']:.4f}"
    )
    print(
        "[過擬合檢查] "
        f"gap={selected_strategy['train_vs_oos_gap']:.4f}, "
        f"fold_dispersion={selected_strategy['fold_dispersion']:.4f}, "
        f"regime_dispersion={selected_strategy['regime_dispersion']:.4f}, "
        f"overfit={bool(selected_strategy['is_overfit'])}"
    )
    print(f"[最佳版本] {selected_strategy['version_id']}")
    print(f"[正式預測版本] {selected_strategy['version_id']}")

    strategy_payload = {
        "selected_strategy": selected_strategy,
        "fallback_strategy": baseline,
    }
    save_json(MODELS_DIR / "strategy_config.json", strategy_payload)

    metadata = {
        "model_type": "catboost",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_rows": int(len(feature_df)),
        "feature_count": len(feature_columns),
        "train_issue_start": int(feature_df["issue"].min()),
        "train_issue_end": int(feature_df["target_issue"].max()),
        "feature_columns_path": "models/feature_columns.json",
        "model_path": "models/catboost_top20.cbm",
        "params": params,
        "selected_strategy": selected_strategy,
        "fallback_strategy": baseline,
        "feature_version": feature_version,
        "runtime_config": {
            "core_windows": cfg.get("core_windows", {}),
            "smoothing_alpha": cfg.get("smoothing_alpha", 0.5),
            "decay_half_lives": cfg.get("decay_half_lives", {}),
            "distance_kernel_tau": cfg.get("distance_kernel_tau", 2),
            "soft_label_training": cfg.get("soft_label_training", {}),
            "proximity_model": cfg.get("proximity_model", {}),
        },
        "soft_label_model_path": (
            "models/catboost_soft_label.cbm"
            if bool(soft_cfg.get("enabled", False))
            else ""
        ),
        "pm1_model_path": (
            "models/catboost_pm1_proximity.cbm"
            if bool(proximity_cfg.get("enabled", False))
            else ""
        ),
        "soft_label_training": cfg.get("soft_label_training", {}),
        "proximity_model": cfg.get("proximity_model", {}),
        "soft_label_normalization_method": str(
            soft_cfg.get("normalization", "rank_pct")
        ),
        "train_rows_used": {
            "exact": int(len(y_train)),
            "soft": int(len(x_soft)) if bool(soft_cfg.get("enabled", False)) else 0,
            "proximity": (
                int(len(x_pm1)) if bool(proximity_cfg.get("enabled", False)) else 0
            ),
        },
    }
    if len(feature_columns) != len(V3_CORE20_COLUMNS):
        raise ValueError(
            f"v3_core20 metadata requires feature_count={len(V3_CORE20_COLUMNS)}"
        )

    save_json(MODELS_DIR / "metadata.json", metadata)


if __name__ == "__main__":
    main()
