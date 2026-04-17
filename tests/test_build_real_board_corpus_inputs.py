from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def test_build_real_board_corpus_reads_jsonl_and_parquet(tmp_path: Path) -> None:
    jsonl = tmp_path / "boards.jsonl"
    jsonl.write_text(json.dumps({"board_id": "j1", "grid": [[1, 2], [3, 4]]}) + "\n", encoding="utf-8")

    parquet = tmp_path / "boards.parquet"
    pd.DataFrame([{"board_id": "p1", "grid": [[1, 2, 3], [4, 5, 6]]}]).to_parquet(parquet, index=False)

    out = tmp_path / "full.jsonl"
    partial = tmp_path / "partial.jsonl"
    audit = tmp_path / "audit.json"

    subprocess.run(
        [
            "python",
            "scripts/build_real_board_corpus.py",
            "--input-dir",
            str(tmp_path),
            "--glob",
            "*",
            "--output",
            str(out),
            "--partial-meta",
            str(partial),
            "--audit",
            str(audit),
        ],
        check=True,
    )

    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = {r["board_id"] for r in rows}
    assert "j1" in ids
    assert "p1" in ids
