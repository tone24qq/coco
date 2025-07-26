import os

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from model import DynamicMET

app = FastAPI()


class BoardInput(BaseModel):
    board: list[list[int]]
    target_value: int


models: dict[tuple[int, int], DynamicMET] = {}


@app.on_event("startup")
def load_models() -> None:
    """Load pretrained models if available, else use fallback."""
    for rows, cols, path in [(8, 10, "met_8x10.pth")]:
        model = DynamicMET(rows * cols, 80)
        if torch is not None and os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
            if hasattr(model, "eval"):
                model.eval()
        models[(rows, cols)] = model


@app.post("/predict")
async def predict(input: BoardInput):
    board = np.array(input.board)
    rows, cols = board.shape
    flat = board.flatten()
    flat = np.where(flat < 0, 0, flat)
    model = models[(rows, cols)]
    if torch is not None:
        inp = torch.tensor(flat).long().unsqueeze(0)
        logits = model(inp)  # type: ignore[misc]
        probs = torch.softmax(logits, dim=-1)
        scores = probs[0, :, input.target_value]
        topk = torch.topk(scores, k=3).indices.tolist()
        scores_np = scores.detach().cpu().numpy()
    else:
        inp = flat.reshape(1, -1)
        logits = model(inp)
        scores_np = logits[0, :, input.target_value]
        topk = np.argsort(scores_np)[-3:][::-1].tolist()
    return [
        {"row": idx // cols, "col": idx % cols, "score": float(scores_np[idx])}
        for idx in topk
    ]
