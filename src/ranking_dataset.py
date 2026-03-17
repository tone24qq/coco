from __future__ import annotations

import numpy as np
import pandas as pd

RANKER_TRAINING_COLUMNS_PREFIX = ["issue", "number"]
RANKER_TRAINING_COLUMNS_SUFFIX = ["label", "group_id"]


def build_ranker_training_rows(
    issue_payloads: dict[int, dict[str, object]],
    indices: list[int],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for idx in indices:
        payload = issue_payloads[int(idx)]
        cand = payload["cand"].copy().reset_index(drop=True)
        if "number" not in cand.columns:
            cand.insert(0, "number", np.arange(1, 81, dtype=int))
        issue_row = payload.get("issue_row")
        issue = (
            int(issue_row.get("issue", idx)) if hasattr(issue_row, "get") else int(idx)
        )
        target = set(int(x) for x in payload["target"])

        block = pd.DataFrame(
            {
                "issue": issue,
                "number": cand["number"].astype(int),
                **{
                    col: cand[col].astype(float) if col in cand.columns else 0.0
                    for col in feature_columns
                },
            }
        )
        block["label"] = block["number"].isin(target).astype(int)
        block["group_id"] = issue
        if len(block) != 80:
            raise ValueError(
                f"ranker training rows expect 80 candidates per issue, got {len(block)}"
            )
        rows.append(block)

    rank_df = pd.concat(rows, ignore_index=True)
    ordered_cols = [
        *RANKER_TRAINING_COLUMNS_PREFIX,
        *feature_columns,
        *RANKER_TRAINING_COLUMNS_SUFFIX,
    ]
    rank_df = rank_df.reindex(columns=ordered_cols)
    return rank_df


def split_ranker_training_frame(
    rank_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = rank_df.reindex(columns=feature_columns).astype(float)
    y = rank_df["label"].astype(float).to_numpy(dtype=float)
    group_id = rank_df["group_id"].astype(int).to_numpy(dtype=np.int64)
    return x, y, group_id
