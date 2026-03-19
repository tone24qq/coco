from pathlib import Path

import pytest

from src.artifacts import load_artifacts
from src.predict import _validate_feature_contract
from src.utils import DataContractError


def test_missing_artifact_fail_fast(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(DataContractError):
        load_artifacts(models)


def test_feature_column_mismatch_fail_fast() -> None:
    class FakeArtifacts:
        feature_columns = ["a", "b"]

    with pytest.raises(DataContractError):
        _validate_feature_contract(__import__("pandas").DataFrame({"a": [1]}), FakeArtifacts())
