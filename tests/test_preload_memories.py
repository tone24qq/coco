import app


def test_load_memory_from_npz():
    app.models.clear()
    app.memories.clear()
    model = app._create_model(4, 5)
    if hasattr(model, "eval"):
        model.eval()
    app.models[(4, 5)] = model
    app._load_memory_for_shape(4, 5)
    assert (4, 5) in app.memories
