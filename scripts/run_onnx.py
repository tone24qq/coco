"""Run ONNX model with onnxruntime."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort


def main(model_path: str) -> None:
    tokens = np.zeros((1, 16), dtype=np.int64)
    attn = np.ones_like(tokens, dtype=bool)
    sess = ort.InferenceSession(model_path)
    logits = sess.run(None, {"tokens": tokens, "attn_mask": attn})[0]
    print(logits.shape)


if __name__ == "__main__":  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    args = parser.parse_args()
    main(args.model)
