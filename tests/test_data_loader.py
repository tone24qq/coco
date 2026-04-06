from pathlib import Path

from src.data_loader import load_and_validate


def test_load_and_validate_sample_data() -> None:
    samples, audit = load_and_validate(Path("samples/data/position_samples.json"))
    assert audit.total_samples >= audit.valid_samples
    assert audit.valid_samples > 0
    assert len(samples) == audit.valid_samples
