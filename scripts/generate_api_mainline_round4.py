from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.image_parse_api import app


CASES = [
    ("gogo/20/IS23120130.jpg", "5x4"),
    ("gogo/80/NL230505019.jpg", "10x8"),
    ("gogo/120/NG230516001_頁面_1.jpg", "12x10"),
]


def _flatten(grid):
    return [v for row in grid for v in row]


def main() -> None:
    client = TestClient(app)
    report = []
    for image, expected_shape in CASES:
        with Path(image).open("rb") as f:
            resp = client.post("/board/parse", files={"image": (Path(image).name, f, "image/jpeg")}, data={"strict": "true"})
        item = {
            "image": image,
            "expected_shape": expected_shape,
            "api_status_code": resp.status_code,
        }
        if resp.status_code != 200:
            report.append(item)
            continue
        body = resp.json()
        numbers = body.get("numbers_all", [])
        legal_max = int(body.get("rows", 0) * body.get("cols", 0))
        illegal = sorted([x for x in numbers if not (1 <= int(x) <= legal_max)])
        duplicates = sorted({x for x in numbers if numbers.count(x) > 1})
        missing = body.get("missing_values", [])
        item.update(
            {
                "shape": body.get("shape"),
                "shape_correct": body.get("shape") == expected_shape,
                "grid_complete_rate": round(1.0 - (len(missing) / max(1, legal_max)), 4),
                "numbers_all_valid": len(illegal) == 0 and len(duplicates) == 0,
                "duplicate_values": duplicates,
                "illegal_values": illegal,
                "missing_values": missing,
                "overlay_returned": bool(body.get("overlay_image_base64") or body.get("overlay_path")),
                "parse_backend_summary": body.get("parse_diagnostics", {}),
            }
        )
        report.append(item)

    baseline_old = {"mean_complete_rate": 0.4292, "numbers_all_valid_count": 1}
    complete_rates = [float(x.get("grid_complete_rate", 0.0)) for x in report if x.get("api_status_code") == 200]
    valid_count = sum(1 for x in report if x.get("numbers_all_valid"))
    dup_count = sum(len(x.get("duplicate_values", [])) for x in report)
    ill_count = sum(len(x.get("illegal_values", [])) for x in report)
    miss_count = sum(len(x.get("missing_values", [])) for x in report)

    summary = {
        "cases": len(report),
        "shape_accuracy": round(sum(1 for x in report if x.get("shape_correct")) / max(1, len(report)), 4),
        "mean_complete_rate": round(sum(complete_rates) / max(1, len(complete_rates)), 4),
        "overlay_rate": round(sum(1 for x in report if x.get("overlay_returned")) / max(1, len(report)), 4),
        "numbers_all_valid_count": valid_count,
        "old_vs_new": {
            "old_mean_complete_rate": baseline_old.get("mean_complete_rate"),
            "new_mean_complete_rate": round(sum(complete_rates) / max(1, len(complete_rates)), 4),
            "old_numbers_all_valid_count": baseline_old.get("numbers_all_valid_count"),
            "new_numbers_all_valid_count": valid_count,
        },
        "stats_diff": {
            "duplicate_values_total": dup_count,
            "illegal_values_total": ill_count,
            "missing_values_total": miss_count,
        },
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/api_mainline_e2e_round4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("reports/api_mainline_e2e_round4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
