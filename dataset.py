from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc]
    TORCH_AVAILABLE = False

MASK_TOKEN_ID = 0
BLANK_VALUE = -1


def validate_board(board: np.ndarray) -> None:
    """Validate board contents.

    Ensures numbers are unique and within ``1..N`` where ``N`` is the board
    size. ``BLANK_VALUE`` (-1) is allowed to denote empty cells.
    """
    n = board.size
    valid_values = set(range(1, n + 1))
    flat = board.ravel()
    for v in flat:
        if v != BLANK_VALUE and v not in valid_values:
            raise ValueError("board values out of range")
    non_blank = flat[flat != BLANK_VALUE]
    if non_blank.size != len(set(non_blank.tolist())):
        raise ValueError("board has duplicate numbers")


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
        for b in self.boards:
            validate_board(np.asarray(b))
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:  # noqa: D401
        """Return dataset size."""
        return len(self.boards)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Return masked input, target and original board for ``idx``."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for ScratchCardDataset")

        board_arr = self.boards[idx].flatten()
        board = torch.from_numpy(board_arr).long()
        target = torch.tensor(int(self.targets[idx])).long()
        mask = torch.rand(board.shape) < self.mask_ratio
        mask |= board == BLANK_VALUE
        inp = board.clone()
        inp[mask] = MASK_TOKEN_ID
        orig = torch.where(board == BLANK_VALUE, MASK_TOKEN_ID, board)
        return {
            "input_vals": inp,
            "mask": mask,
            "orig_vals": orig,
            "target": target,
        }
