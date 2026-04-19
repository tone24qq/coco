from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.safe_io import SafeWriteConfig, read_dataset_auto, write_dataframe_chunks_safe


def test_streaming_chunk_writer_roundtrip(tmp_path: Path) -> None:
    chunks = [pd.DataFrame({"a": [1, 2], "b": [3, 4]}), pd.DataFrame({"a": [5], "b": [6]})]
    out = tmp_path / "ds.parquet"
    meta = write_dataframe_chunks_safe(chunks, out, fmt="parquet", config=SafeWriteConfig(producer_script="t"))
    assert meta["type"] == "dataset_dir"
    loaded = read_dataset_auto(out)
    assert len(loaded) == 3
