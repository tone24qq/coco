from src.inference.model_loader import load_model


def test_load_model_handles_empty_file(tmp_path):
    ckpt = tmp_path / "empty.ckpt"
    ckpt.touch()
    model = load_model(ckpt)
    assert hasattr(model, "forward")
