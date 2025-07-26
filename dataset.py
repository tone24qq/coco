from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

MASK_TOKEN_ID = 0


class ScratchCardDataset(Dataset):
    """Dataset that randomly masks board values for training.

    Parameters
    ----------
    boards : list of Tuple[np.ndarray, int]
        List of ``(board, target)`` pairs.
    mask_ratio : float, optional
        Fraction of fields to mask, by default ``0.6``.
    """

    def __init__(
        self, boards: List[Tuple[np.ndarray, int]], mask_ratio: float = 0.6
    ) -> None:
        self.boards, self.targets = zip(*boards)
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:  # noqa: D401
        """Return dataset size."""
        return len(self.boards)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return masked input, target and original board for ``idx``."""
        board = torch.from_numpy(self.boards[idx].flatten()).long()
        target = torch.tensor(int(self.targets[idx])).long()
        mask = torch.rand(board.shape) < self.mask_ratio
        inp = board.clone()
        inp[mask] = MASK_TOKEN_ID
        return {
            "input_vals": inp,
            "mask": mask,
            "orig_vals": board,
            "target": target,
        }
