from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

from winwin_service.config import AppConfig


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    wf = _load_module("scripts/strict_walkforward_search.py", "wf")
    fh = _load_module("scripts/final_holdout_validation.py", "fh")

    out_path = Path("reports/comparison_numeric_tuning.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_summary = json.loads(
        Path("reports/final_holdout/summary_report.json").read_text()
    )
    walk_summary = json.loads(
        Path("reports/walkforward/summary_report.json").read_text()
    )
    tuned_payload = json.loads(
        Path("reports/final_holdout/best_config.json").read_text()
    )
    tuned_params = tuned_payload.get("params", tuned_payload)

    draws_payload = json.loads(
        Path("reports/final_holdout/snapshot_draws.json").read_text()
    )
    draws = [
        (int(p), [int(n) for n in nums]) for p, nums in draws_payload["draws"]
    ]
    periods = [p for p, _ in draws]
    numbers = [d for _, d in draws]

    windows = fh.split_windows(len(draws), 0.6, 0.2, 0.2, min_train_draws=20)
    final_start, final_end = windows["final"]
    blocks = fh.split_final_blocks(final_start, final_end)

    before_cfg = AppConfig()
    after_cfg = fh.build_config(tuned_params)

    _, before_final_buckets = fh.evaluate_window(
        numbers,
        periods,
        final_start,
        final_end,
        before_cfg,
        seed=9101,
        include_baselines=True,
    )
    _, after_final_buckets = fh.evaluate_window(
        numbers,
        periods,
        final_start,
        final_end,
        after_cfg,
        seed=9101,
        include_baselines=True,
    )

    before_final = fh.aggregate(before_final_buckets["model"])
    after_final = fh.aggregate(after_final_buckets["model"])

    block_rows = []
    for i, (bs, be) in enumerate(blocks, start=1):
        _, before_b = fh.evaluate_window(
            numbers,
            periods,
            bs,
            be,
            before_cfg,
            seed=9200 + i,
            include_baselines=True,
        )
        _, after_b = fh.evaluate_window(
            numbers,
            periods,
            bs,
            be,
            after_cfg,
            seed=9200 + i,
            include_baselines=True,
        )
        block_rows.append(
            {
                "block_id": i,
                "before": fh.aggregate(before_b["model"]),
                "after": fh.aggregate(after_b["model"]),
            }
        )

    wf_draws = wf.fetch_period_draws(30)
    if len(wf_draws) > 180:
        wf_draws = wf_draws[-180:]
    wf_periods = [p for p, _ in wf_draws]
    wf_numbers = [d for _, d in wf_draws]
    holdouts = wf.make_holdouts(len(wf_numbers), min_train=12)
    before_wf, _ = wf.eval_config(
        wf_numbers, wf_periods, before_cfg, holdouts, seed=333
    )
    after_wf, _ = wf.eval_config(
        wf_numbers, wf_periods, after_cfg, holdouts, seed=333
    )

    uniform = fh.aggregate(after_final_buckets["uniform_random"])
    frequency = fh.aggregate(after_final_buckets["frequency"])

    changed = {
        k: {"before": getattr(before_cfg, k), "after": v}
        for k, v in tuned_params.items()
        if hasattr(before_cfg, k) and getattr(before_cfg, k) != v
    }

    out = {
        "before_params": asdict(before_cfg),
        "after_params": tuned_params,
        "changed_knobs": changed,
        "walkforward": {
            "before": {
                "same_triplet_2hit_rate": before_wf["same_triplet_2hit_rate"],
                "top1_2hit_rate": before_wf["top1_2hit_rate"],
                "same_triplet_3hit_rate": before_wf["same_triplet_3hit_rate"],
            },
            "after": {
                "same_triplet_2hit_rate": after_wf["same_triplet_2hit_rate"],
                "top1_2hit_rate": after_wf["top1_2hit_rate"],
                "same_triplet_3hit_rate": after_wf["same_triplet_3hit_rate"],
            },
            "report_best_summary": walk_summary["best_summary"],
        },
        "final_holdout": {
            "before": {
                "final_same_triplet_2hit_rate": before_final[
                    "same_triplet_2hit_rate"
                ],
                "final_top1_2hit_rate": before_final["top1_2hit_rate"],
                "final_same_triplet_3hit_rate": before_final[
                    "same_triplet_3hit_rate"
                ],
            },
            "after": {
                "final_same_triplet_2hit_rate": after_final[
                    "same_triplet_2hit_rate"
                ],
                "final_top1_2hit_rate": after_final["top1_2hit_rate"],
                "final_same_triplet_3hit_rate": after_final[
                    "same_triplet_3hit_rate"
                ],
            },
            "blocks": block_rows,
            "report_summary": final_summary,
        },
        "vs_uniform_random": {
            "same_triplet_2hit_rate_delta": after_final[
                "same_triplet_2hit_rate"
            ]
            - uniform["same_triplet_2hit_rate"],
            "top1_2hit_rate_delta": after_final["top1_2hit_rate"]
            - uniform["top1_2hit_rate"],
            "same_triplet_3hit_rate_delta": after_final[
                "same_triplet_3hit_rate"
            ]
            - uniform["same_triplet_3hit_rate"],
        },
        "vs_frequency": {
            "same_triplet_2hit_rate_delta": after_final[
                "same_triplet_2hit_rate"
            ]
            - frequency["same_triplet_2hit_rate"],
            "top1_2hit_rate_delta": after_final["top1_2hit_rate"]
            - frequency["top1_2hit_rate"],
            "same_triplet_3hit_rate_delta": after_final[
                "same_triplet_3hit_rate"
            ]
            - frequency["same_triplet_3hit_rate"],
        },
        "uplift_abs": {
            "same_triplet_2hit_rate": after_final["same_triplet_2hit_rate"]
            - before_final["same_triplet_2hit_rate"],
            "top1_2hit_rate": after_final["top1_2hit_rate"]
            - before_final["top1_2hit_rate"],
            "same_triplet_3hit_rate": after_final["same_triplet_3hit_rate"]
            - before_final["same_triplet_3hit_rate"],
        },
        "uplift_pct": {
            "same_triplet_2hit_rate": (
                0.0
                if before_final["same_triplet_2hit_rate"] == 0
                else (
                    after_final["same_triplet_2hit_rate"]
                    - before_final["same_triplet_2hit_rate"]
                )
                / before_final["same_triplet_2hit_rate"]
            ),
            "top1_2hit_rate": (
                0.0
                if before_final["top1_2hit_rate"] == 0
                else (
                    after_final["top1_2hit_rate"]
                    - before_final["top1_2hit_rate"]
                )
                / before_final["top1_2hit_rate"]
            ),
            "same_triplet_3hit_rate": (
                0.0
                if before_final["same_triplet_3hit_rate"] == 0
                else (
                    after_final["same_triplet_3hit_rate"]
                    - before_final["same_triplet_3hit_rate"]
                )
                / before_final["same_triplet_3hit_rate"]
            ),
        },
        "consistent_improvement": {
            "walkforward_and_final_same_triplet_2hit_rate": (
                after_wf["same_triplet_2hit_rate"]
                >= before_wf["same_triplet_2hit_rate"]
                and after_final["same_triplet_2hit_rate"]
                >= before_final["same_triplet_2hit_rate"]
            )
        },
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
