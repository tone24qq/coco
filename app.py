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
    """Load pretrained models into memory."""
    for rows, cols, path in [(8, 10, "met_8x10.pth")]:
        model = DynamicMET(rows * cols, 80)
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
        model.eval()
        models[(rows, cols)] = model


@app.post("/predict")
async def predict(input: BoardInput):
    board = np.array(input.board)
    rows, cols = board.shape
    flat = board.flatten()
    flat = np.where(flat < 0, 0, flat)
    inp = torch.tensor(flat).long().unsqueeze(0)
    logits = models[(rows, cols)](inp)
    probs = torch.softmax(logits, dim=-1)
    scores = probs[0, :, input.target_value]
    topk = torch.topk(scores, k=3).indices.tolist()
    return [{"row": idx // cols, "col": idx % cols} for idx in topk]
