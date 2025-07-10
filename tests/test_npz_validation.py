import json

import numpy as np

import analyzer


def test_invalid_npz_counter(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    valid_meta = {"samples": 1, "generated_at": "now"}
    meta_bytes = np.frombuffer(json.dumps(valid_meta).encode(), dtype=np.uint8)
    np.savez(samples / "2x2.npz", freq=np.zeros((2, 2, 5), int), meta=meta_bytes)
    invalid_meta = np.array([1, 2, 3], dtype=np.uint8)
    np.savez(samples / "3x3.npz", freq=np.zeros((3, 3, 10), int), meta=invalid_meta)
    analyzer.get_sample_stats_cached.cache_clear()
    before = analyzer.INVALID_NPZ_COUNTER._value.get()
    analyzer.get_sample_stats_cached(2, 2, str(samples))
    analyzer.get_sample_stats_cached(3, 3, str(samples))
    after = analyzer.INVALID_NPZ_COUNTER._value.get()
    assert after - before == 1
    assert analyzer.get_sample_stats_cached.cache_info().currsize <= 6
