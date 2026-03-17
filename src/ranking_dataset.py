from __future__ import annotations

import numpy as np
import pandas as pd


def build_ranker_training_rows(
    issue_payloads: dict[int, dict[str, object]],
    indices: list[int],
    feature_columns: list[str],
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for idx in indices:
        payload = issue_payloads[int(idx)]
        cand = payload["cand"].copy().reset_index(drop=True)
        if "number" not in cand.columns:
            cand.insert(0, "number", np.arange(1, 81, dtype=int))
        issue = int(payload["issue_row"]["issue"])
        target = set(int(x) for x in payload["target"])
        block = pd.DataFrame(
            {
                "issue": issue,
                "number": cand["number"].astype(int),
                **{col: cand[col].astype(float) for col in feature_columns},
            }
        )
        block["label"] = block["number"].isin(target).astype(int)
        block["group_id"] = issue
        if len(block) != 80:
            raise ValueError(f"ranker group must have 80 rows, got {len(block)}")
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    required_columns = ["issue", "number", *feature_columns, "label", "group_id"]
    return out.reindex(columns=required_columns)


def split_ranker_training_frame(
    rank_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = rank_df.reindex(columns=feature_columns).copy()
    y = rank_df["label"].astype(float).to_numpy()
    group_id = rank_df["group_id"].astype(int).to_numpy()
    return x, y, group_id


def build_ranker_training_frame(
    issue_payloads: dict[int, dict[str, object]],
    indices: list[int],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rank_df = build_ranker_training_rows(issue_payloads, indices, feature_columns)
    return (
        rank_df[["issue", "number", *feature_columns, "group_id"]].copy(),
        rank_df["label"].astype(int),
        rank_df["group_id"].astype(int),
    )
