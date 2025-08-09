import onnx
import torch

from src.inference.model_loader import load_model


def test_export_onnx(tmp_path):
    model = load_model("weights/best.ckpt")
    model.eval()
    tokens = torch.zeros(1, 16, dtype=torch.long)
    attn = torch.ones_like(tokens, dtype=torch.bool)
    out = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        (tokens, attn),
        out,
        opset_version=13,
        input_names=["tokens", "attn_mask"],
        output_names=["logits"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "tokens"},
            "attn_mask": {0: "batch", 1: "tokens"},
            "logits": {0: "batch", 1: "tokens"},
        },
    )
    assert out.exists()
    onnx.load(str(out))
