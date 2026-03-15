from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from catboost import CatBoostClassifier


@dataclass
class CascadeArtifacts:
    pipeline_version: str
    stage1_model: CatBoostClassifier
    stage2_model: CatBoostClassifier
    stage1_feature_columns: list[str]
    stage2_feature_columns: list[str]
    stage3_input_schema: list[str]
    stage1_keep: int
    stage2_keep: int


def save_cascade_artifacts(
    base_dir: Path,
    artifacts: CascadeArtifacts,
    feature_version: str,
    train_issue_start: int,
    train_issue_end: int,
) -> dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    stage1_model_path = base_dir / "stage1_model.cbm"
    stage2_model_path = base_dir / "stage2_model.cbm"
    stage1_cols_path = base_dir / "stage1_feature_columns.json"
    stage2_cols_path = base_dir / "stage2_feature_columns.json"
    stage3_schema_path = base_dir / "stage3_input_schema.json"
    metadata_path = base_dir / "pipeline_metadata.json"

    artifacts.stage1_model.save_model(str(stage1_model_path))
    artifacts.stage2_model.save_model(str(stage2_model_path))
    stage1_cols_path.write_text(
        json.dumps(artifacts.stage1_feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage2_cols_path.write_text(
        json.dumps(artifacts.stage2_feature_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage3_schema_path.write_text(
        json.dumps(artifacts.stage3_input_schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "pipeline_version": artifacts.pipeline_version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": feature_version,
        "stage1_keep": int(artifacts.stage1_keep),
        "stage2_keep": int(artifacts.stage2_keep),
        "stage1_model_path": stage1_model_path.name,
        "stage2_model_path": stage2_model_path.name,
        "stage1_columns_path": stage1_cols_path.name,
        "stage2_columns_path": stage2_cols_path.name,
        "stage3_input_schema_path": stage3_schema_path.name,
        "train_issue_start": int(train_issue_start),
        "train_issue_end": int(train_issue_end),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def load_cascade_artifacts(base_dir: Path) -> CascadeArtifacts:
    metadata_path = base_dir / "pipeline_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    stage1_model = CatBoostClassifier()
    stage1_model.load_model(str(base_dir / metadata["stage1_model_path"]))
    stage2_model = CatBoostClassifier()
    stage2_model.load_model(str(base_dir / metadata["stage2_model_path"]))

    stage1_feature_columns = json.loads(
        (base_dir / metadata["stage1_columns_path"]).read_text(encoding="utf-8")
    )
    stage2_feature_columns = json.loads(
        (base_dir / metadata["stage2_columns_path"]).read_text(encoding="utf-8")
    )
    stage3_input_schema = json.loads(
        (base_dir / metadata["stage3_input_schema_path"]).read_text(encoding="utf-8")
    )

    return CascadeArtifacts(
        pipeline_version=str(metadata.get("pipeline_version", "cascade_v1")),
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage1_feature_columns=list(stage1_feature_columns),
        stage2_feature_columns=list(stage2_feature_columns),
        stage3_input_schema=list(stage3_input_schema),
        stage1_keep=int(metadata.get("stage1_keep", 30)),
        stage2_keep=int(metadata.get("stage2_keep", 10)),
    )


def ensure_columns(
    df_columns: Sequence[str], expected: Sequence[str], stage: str
) -> None:
    if list(df_columns) != list(expected):
        raise ValueError(f"{stage} feature columns mismatch")
