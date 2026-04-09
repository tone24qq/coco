from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scripts.run_hit_benchmark import BenchmarkCase, run_benchmark
from src.api import app
from src.inference_facade import infer_target_position
from src.inference_service import run_inference


def _sample_board() -> list[list[int]]:
    return [[1, -1, 3], [-1, 5, -1]]


def test_api_and_facade_and_service_consistent_best_cell() -> None:
    board = _sample_board()
    target = 4

    service = run_inference(board, target, source="service")
    facade = infer_target_position(board, target, source="facade")
    client = TestClient(app)
    api_resp = client.post("/infer_target_position", json={"board": board, "target_number": target})

    assert api_resp.status_code == 200
    api = api_resp.json()

    assert service["best_cell"]["row"] == facade["best_cell"]["row"] == api["best_cell"]["row"]
    assert service["best_cell"]["col"] == facade["best_cell"]["col"] == api["best_cell"]["col"]
    assert [c["row"] for c in service["candidate_cells"]] == [c["row"] for c in api["candidate_cells"]]


def test_contract_metadata_non_probability() -> None:
    result = run_inference(_sample_board(), 4, source="test")
    md = result["metadata"]
    assert md["confidence_1_to_100_is_probability"] is False
    assert "non_calibrated" in md["confidence_1_to_100_type"]


def test_benchmark_outputs_and_schema(tmp_path: Path) -> None:
    case = BenchmarkCase(
        sample_id="s1",
        size_class="6",
        source="unit",
        full_board=[[1, 2, 3], [4, 5, 6]],
        masked_board=[[1, -1, 3], [-1, 5, -1]],
        target_number=4,
        true_cell_0_based=(1, 0),
    )
    out = run_benchmark([case], output_dir=tmp_path, seed=123)
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "per_case_predictions.csv").exists()
    assert (tmp_path / "error_cases.json").exists()
    assert (tmp_path / "bottleneck_report.json").exists()
    assert (tmp_path / "ablation_summary.json").exists()

    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert "strategy_summaries" in summary
    assert "full_fusion" in summary["strategy_summaries"]
    assert "top5_hit_rate" in summary["strategy_summaries"]["full_fusion"]
    assert "random_baseline" in out["ablation_summary"]
