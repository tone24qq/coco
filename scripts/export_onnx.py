"""Export a trained model to ONNX."""

import argparse

import torch

from src.inference.model_loader import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("out", type=str)
    args = parser.parse_args()
    model = load_model(args.ckpt)
    model.eval()
    dummy = torch.zeros(1, 4 * 4, dtype=torch.long)
    torch.onnx.export(model, (dummy, None, 16), args.out)


if __name__ == "__main__":
    main()
