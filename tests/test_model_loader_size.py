from src.inference import model_loader


def test_get_weight_path(tmp_path):
    (tmp_path / "best.ckpt").touch()
    specific = tmp_path / "3x3.ckpt"
    specific.touch()
    assert model_loader.get_weight_path(3, 3, tmp_path) == specific
    assert model_loader.get_weight_path(4, 4, tmp_path) == tmp_path / "best.ckpt"


def test_load_model_for_size_caches(tmp_path):
    model_loader.MODEL_CACHE.clear()
    (tmp_path / "best.ckpt").touch()
    m1 = model_loader.load_model_for_size(4, 4, base=tmp_path)
    m2 = model_loader.load_model_for_size(4, 4, base=tmp_path)
    assert m1 is m2
