import json

from src.predict import Predictor
from src.utils import MODELS_DIR, V3_CORE20_COLUMNS


class _DummyModel:
    def load_model(self, _path: str) -> None:
        return None


def test_predictor_runtime_config_prefers_metadata(monkeypatch) -> None:
    (MODELS_DIR / "feature_columns.json").write_text(
        json.dumps(V3_CORE20_COLUMNS), encoding="utf-8"
    )
    runtime_cfg = {
        "feature_version": "v3_core20",
        "core_windows": {"z_window": 77},
        "smoothing_alpha": 0.9,
        "decay_half_lives": {"ewma": 12},
        "distance_kernel_tau": 9,
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps({"feature_version": "v3_core20", "runtime_config": runtime_cfg}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.predict.CatBoostClassifier", _DummyModel)
    predictor = Predictor.load()
    assert predictor.runtime_config["core_windows"]["z_window"] == 77
    assert predictor.runtime_config["distance_kernel_tau"] == 9
