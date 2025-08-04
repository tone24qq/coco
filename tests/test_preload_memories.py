import app


def test_load_memory_prefers_json():
    app.models.clear()
    app.memories.clear()
    app.memory_files.clear()
    model = app._create_model(4, 5)
    if hasattr(model, "eval"):
        model.eval()
    app.models[(4, 5)] = model
    app._load_memory_for_shape(4, 5, model)
    assert (4, 5) in app.memories
    assert (4, 5) not in app.memory_files
