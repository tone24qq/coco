from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.analysis.features import build_local_analysis_bundle


def run_analysis_engine(
    recent_draws: Sequence[Sequence[int]],
    *,
    snapshot: pd.DataFrame | None = None,
) -> dict:
    bundle = build_local_analysis_bundle(recent_draws)
    summary = {
        "engine_version": "v1_local_history",
        "sample_size": len(recent_draws),
        "uses_snapshot": snapshot is not None,
    }
    if snapshot is not None and not snapshot.empty:
        summary["snapshot_rows"] = int(len(snapshot))
        summary["snapshot_types"] = (
            snapshot.get("snapshot_type", pd.Series(dtype=str)).value_counts().to_dict()
        )
    return {
        "summary": summary,
        **bundle,
    }
