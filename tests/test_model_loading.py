from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from rf_infer.core import _extract_features, _predict_proba_any

MODELS_DIR = Path("models")


def _dummy_board():
    # 4x5 board with target=7
    return (
        np.array(
            [
                [1, -1, 3, 4, 5],
                [6, -1, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ],
            dtype=int,
        ),
        7,
    )


def test_model_file_exists():
    assert MODELS_DIR.exists(), "models/ 資料夾不存在"
    files = (
        list(MODELS_DIR.glob("*.pkl"))
        + list(MODELS_DIR.glob("*.txt"))
        + list(MODELS_DIR.glob("*.bin"))
    )
    assert files, "models/ 裡面找不到任何模型檔"


def test_model_can_load_and_predict_proba():
    board, target = _dummy_board()
    X = _extract_features(board, target)

    for f in MODELS_DIR.iterdir():
        if f.suffix not in {".pkl", ".txt", ".bin"}:
            continue
        model = joblib.load(f) if f.suffix == ".pkl" else lgb.Booster(model_file=str(f))
        probs = _predict_proba_any(model, X)
        assert probs.shape[0] == X.shape[0], f"{f.name} 機率輸出筆數錯誤"
        assert probs.shape[1] in (
            2,
            120,
        ), f"{f.name} 機率輸出維度不合理: {probs.shape}"
        assert np.all((probs >= 0) & (probs <= 1)), f"{f.name} 機率值超出 0~1"
