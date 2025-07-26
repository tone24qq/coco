from typing import Dict

import torch
from torch.utils.data import Dataset

MASK_TOKEN_ID = 0


class ScratchCardDataset(Dataset):
    """Dataset that randomly masks board values for training."""

    def __init__(self, boards: list, mask_ratio: float = 0.6) -> None:
        self.boards = boards
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:  # noqa: D401
        """Return dataset size."""
        return len(self.boards)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return masked input and original board for index ``idx``."""
        board = torch.from_numpy(self.boards[idx].flatten()).long()
        mask = torch.rand(board.shape) < self.mask_ratio
        inp = board.clone()
        inp[mask] = MASK_TOKEN_ID
        return {"input_vals": inp, "mask": mask, "orig_vals": board}
