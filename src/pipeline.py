from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.artifacts import CascadeArtifacts, ensure_columns
from src.selector import build_selector_context, select_top3_combination
from src.utils import (
    CASCADE_V1_STAGE1_COLUMNS,
    CASCADE_V1_STAGE2_COLUMNS,
    CASCADE_V1_STAGE3_COLUMNS,
    build_stage1_candidate_matrix,
    build_stage2_candidate_matrix,
    build_stage3_selector_inputs,
)


@dataclass(frozen=True)
class Stage1RecallGate:
    model: CatBoostClassifier
    feature_columns: list[str]
    keep_k: int

    def predict(self, issue_row: pd.Series) -> pd.DataFrame:
        stage1 = build_stage1_candidate_matrix(issue_row, self.feature_columns)
        ensure_columns(stage1.columns[1:], self.feature_columns, "stage1")
        x = stage1[self.feature_columns]
        scores = self.model.predict_proba(x)[:, 1]
        stage1["stage1_score"] = scores
        stage1 = stage1.sort_values("stage1_score", ascending=False).reset_index(
            drop=True
        )
        stage1["stage1_rank"] = np.arange(1, len(stage1) + 1, dtype=int)
        stage1["stage1_keep_flag"] = (stage1["stage1_rank"] <= int(self.keep_k)).astype(
            int
        )
        return stage1


@dataclass(frozen=True)
class Stage2RegimeRanker:
    model: CatBoostClassifier
    feature_columns: list[str]
    keep_k: int

    def predict(self, issue_row: pd.Series, stage1_df: pd.DataFrame) -> pd.DataFrame:
        stage2 = build_stage2_candidate_matrix(
            issue_row, stage1_df, self.feature_columns
        )
        ensure_columns(
            [c for c in stage2.columns if c in self.feature_columns],
            self.feature_columns,
            "stage2",
        )
        scores = self.model.predict_proba(stage2[self.feature_columns])[:, 1]
        stage2["stage2_score"] = scores
        stage2 = stage2.sort_values("stage2_score", ascending=False).reset_index(
            drop=True
        )
        stage2["stage2_rank"] = np.arange(1, len(stage2) + 1, dtype=int)
        stage2["stage2_keep_flag"] = (stage2["stage2_rank"] <= int(self.keep_k)).astype(
            int
        )
        return stage2


@dataclass(frozen=True)
class Stage3SelectorInputBuilder:
    columns: list[str]

    def build(
        self, issue_row: pd.Series, stage2_df: pd.DataFrame, top_k: int
    ) -> pd.DataFrame:
        out = build_stage3_selector_inputs(issue_row, stage2_df, top_k=top_k)
        ensure_columns(out.columns, self.columns, "stage3_inputs")
        return out


class CascadePipeline:
    @staticmethod
    def _sanitize_issue_row(issue_row: pd.Series) -> pd.Series:
        blocked = {
            "target_numbers",
            "target_issue",
            "actual",
            "label",
            "y",
            "future_numbers",
            "future_issue",
        }
        keep = [c for c in issue_row.index if c not in blocked]
        return issue_row[keep].copy()

    def __init__(
        self,
        stage1: Stage1RecallGate,
        stage2: Stage2RegimeRanker,
        stage3: Stage3SelectorInputBuilder,
    ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3

    @classmethod
    def from_artifacts(cls, artifacts: CascadeArtifacts) -> "CascadePipeline":
        return cls(
            stage1=Stage1RecallGate(
                model=artifacts.stage1_model,
                feature_columns=artifacts.stage1_feature_columns,
                keep_k=artifacts.stage1_keep,
            ),
            stage2=Stage2RegimeRanker(
                model=artifacts.stage2_model,
                feature_columns=artifacts.stage2_feature_columns,
                keep_k=artifacts.stage2_keep,
            ),
            stage3=Stage3SelectorInputBuilder(columns=artifacts.stage3_input_schema),
        )

    @classmethod
    def train(
        cls,
        feature_df: pd.DataFrame,
        stage1_keep: int,
        stage2_keep: int,
        catboost_params: dict,
    ) -> tuple["CascadePipeline", CascadeArtifacts]:
        stage1_x_rows: list[dict] = []
        stage1_y: list[int] = []
        for _, issue_row in feature_df.iterrows():
            s1 = build_stage1_candidate_matrix(issue_row, CASCADE_V1_STAGE1_COLUMNS)
            target = set(json.loads(str(issue_row["target_numbers"])))
            for _, row in s1.iterrows():
                rec = row[CASCADE_V1_STAGE1_COLUMNS].to_dict()
                stage1_x_rows.append(rec)
                stage1_y.append(1 if int(row["number"]) in target else 0)

        stage1_x = pd.DataFrame(stage1_x_rows)
        stage1_model = CatBoostClassifier(**catboost_params)
        stage1_model.fit(stage1_x, stage1_y, verbose=False)

        stage2_x_rows: list[dict] = []
        stage2_y: list[int] = []
        stage1_gate = Stage1RecallGate(
            model=stage1_model,
            feature_columns=list(CASCADE_V1_STAGE1_COLUMNS),
            keep_k=int(stage1_keep),
        )
        for _, issue_row in feature_df.iterrows():
            stage1_df = stage1_gate.predict(issue_row)
            s2 = build_stage2_candidate_matrix(
                issue_row, stage1_df, CASCADE_V1_STAGE2_COLUMNS
            )
            target = set(json.loads(str(issue_row["target_numbers"])))
            for _, row in s2.iterrows():
                rec = row[CASCADE_V1_STAGE2_COLUMNS].to_dict()
                stage2_x_rows.append(rec)
                stage2_y.append(1 if int(row["number"]) in target else 0)

        stage2_x = pd.DataFrame(stage2_x_rows)
        stage2_model = CatBoostClassifier(**catboost_params)
        stage2_model.fit(stage2_x, stage2_y, verbose=False)

        artifacts = CascadeArtifacts(
            pipeline_version="cascade_v1",
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            stage1_feature_columns=list(CASCADE_V1_STAGE1_COLUMNS),
            stage2_feature_columns=list(CASCADE_V1_STAGE2_COLUMNS),
            stage3_input_schema=list(CASCADE_V1_STAGE3_COLUMNS),
            stage1_keep=int(stage1_keep),
            stage2_keep=int(stage2_keep),
        )
        pipeline = cls.from_artifacts(artifacts)
        return pipeline, artifacts

    def predict_issue(self, issue_row: pd.Series) -> dict[str, object]:
        safe_issue_row = self._sanitize_issue_row(issue_row)
        stage1_df = self.stage1.predict(safe_issue_row)
        stage2_df = self.stage2.predict(safe_issue_row, stage1_df)
        stage3_inputs = self.stage3.build(
            safe_issue_row, stage2_df, top_k=self.stage2.keep_k
        )
        stage3_inputs = stage3_inputs.merge(
            stage2_df[["number", "stage1_score"]],
            on="number",
            how="left",
        )
        retained_top10 = set(
            int(x)
            for x in stage2_df[stage2_df["stage2_keep_flag"] == 1]["number"].tolist()
        )
        stage3_numbers = set(int(x) for x in stage3_inputs["number"].tolist())
        if not stage3_numbers.issubset(retained_top10):
            raise ValueError(
                "leakage guard: stage3 inputs must come from stage2 retained top10"
            )

        no_selector_top3 = (
            stage2_df.sort_values("stage2_score", ascending=False)["number"]
            .head(3)
            .astype(int)
            .tolist()
        )
        selector_context = build_selector_context(safe_issue_row)
        selector_result = select_top3_combination(stage3_inputs, selector_context)
        final_top3 = selector_result.final_top3

        final_scores = np.full(80, -1e9, dtype=float)
        stage1_lookup = {
            int(r["number"]): float(r["stage1_score"]) for _, r in stage1_df.iterrows()
        }
        for n in range(1, 81):
            final_scores[n - 1] = stage1_lookup.get(n, -1e9) - 2.0
        for _, row in stage2_df.iterrows():
            n = int(row["number"])
            final_scores[n - 1] = float(row["stage2_score"]) - 1.0
            if int(row["stage2_keep_flag"]) == 1:
                final_scores[n - 1] = float(row["stage2_score"])
        selector_boost = 0.3
        for n in final_top3:
            final_scores[int(n) - 1] += selector_boost

        return {
            "final_scores": final_scores,
            "stage1": stage1_df,
            "stage2": stage2_df,
            "stage3_inputs": stage3_inputs,
            "final_top3": final_top3,
            "no_selector_top3": no_selector_top3,
            "selector_score": selector_result.selector_score,
            "selector_reason": selector_result.reason,
            "selector_table": selector_result.scored_table,
            "selector_regime": selector_context.regime,
        }
