import json

import numpy as np

import analyzer


def test_load_sample_stats_from_full_stats(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    freq = np.zeros((2, 2, 5), dtype=int)
    meta = {"samples": 1, "generated_at": "now"}
    meta_bytes = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(samples / "full_stats_2x2.npz", freq=freq, meta=meta_bytes)

    analyzer.get_sample_stats_cached.cache_clear()
    arr = analyzer.get_sample_stats_cached(2, 2, str(samples))
    assert arr is not None
    assert arr.shape == freq.shape
