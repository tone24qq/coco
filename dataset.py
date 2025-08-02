from typing import Dict, List, Sequence, Tuple, Union

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
# Maximum board value is derived from board size at runtime


def mask_target_patch(
    board: np.ndarray, target_pos: tuple[int, int], patch_size: int = 3
) -> np.ndarray:
    """Return a boolean mask for ``board`` masking a square patch around target."""
    rows, cols = board.shape
    half = patch_size // 2
    r0, c0 = target_pos
    mask = np.zeros_like(board, dtype=bool)
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < rows and 0 <= c < cols:
                mask[r, c] = True
    return mask


def validate_board(
    board: np.ndarray,
    *,
    allow_blank: bool = True,
    require_complete: bool = False,
) -> None:
    """Validate board contents using dynamic dimensions.

    Parameters
    ----------
    board:
        2D array representing the board. ``BLANK_VALUE`` denotes empty cells.
    allow_blank:
        Whether ``BLANK_VALUE`` is permitted in ``board``.
    require_complete:
        When ``True`` the board must contain every number from ``1`` to
        ``rows*cols`` exactly once with **no** blanks.
    """

    rows, cols = board.shape
    num_fields = rows * cols
    flat = board.ravel()
    mask = flat != BLANK_VALUE
    if not allow_blank and np.any(~mask):
        raise ValueError("board contains blank values")

    non_blank = flat[mask]
    if np.any(non_blank < 1) or np.any(non_blank > num_fields):
        raise ValueError("board values out of range")
    if non_blank.size != len(set(non_blank.tolist())):
        raise ValueError("board has duplicate numbers")
    if require_complete:
        if non_blank.size != num_fields:
            raise ValueError("board must contain numbers 1..N exactly once")
        if set(non_blank.tolist()) != set(range(1, num_fields + 1)):
            raise ValueError("board must be a permutation of 1..N")


class ScratchCardDataset(Dataset):
    """Dataset that masks board values according to a chosen policy.

    Parameters
    ----------
    boards : list of Tuple[np.ndarray, int]
        List of ``(board, target)`` pairs.
    mask_ratio : float or Sequence[float], optional
        Used only when ``mode`` is ``"reconstruct"``. Fraction of fields to
        mask. When a sequence ``(min, max)`` is given, a ratio is uniformly
        sampled from that range for each item. Defaults to ``0.6``.
    mode : {"target", "reconstruct"}, optional
        ``"target"`` masks only the target cell. ``"reconstruct"`` masks a
        random fraction of cells and always includes the target. Defaults to
        ``"reconstruct"``.
    """

    def __init__(
        self,
        boards: List[Tuple[np.ndarray, int]],
        mask_ratio: Union[float, Sequence[float]] = 0.6,
        *,
        mode: str = "reconstruct",
        patch_size: int = 3,
    ) -> None:
        self.boards, self.targets = zip(*boards)
        for b in self.boards:
            validate_board(np.asarray(b), allow_blank=False, require_complete=True)

        if mode not in {"target", "reconstruct", "patch"}:
            raise ValueError("mode must be 'target', 'patch' or 'reconstruct'")
        self.mode = mode
        self.patch_size = int(patch_size)

        if self.mode == "reconstruct":
            if isinstance(mask_ratio, Sequence) and not isinstance(
                mask_ratio, (bytes, str)
            ):
                if len(mask_ratio) != 2:
                    raise ValueError("mask_ratio range must have two elements")
                low, high = float(mask_ratio[0]), float(mask_ratio[1])
                if not 0.0 <= low <= high <= 1.0:
                    raise ValueError("mask_ratio values must be between 0 and 1")
                self.mask_ratio = None
                self.mask_ratio_range = (low, high)
            else:
                ratio = float(mask_ratio)
                if not 0.0 <= ratio <= 1.0:
                    raise ValueError("mask_ratio must be between 0 and 1")
                self.mask_ratio = ratio
                self.mask_ratio_range = None
        else:
            self.mask_ratio = 0.0
            self.mask_ratio_range = None

    def __len__(self) -> int:  # noqa: D401
        """Return dataset size."""
        return len(self.boards)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        """Return masked input, target and original board for ``idx``."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for ScratchCardDataset")

        board_arr = self.boards[idx].flatten()
        board_arr_mapped = np.where(board_arr == BLANK_VALUE, MASK_TOKEN_ID, board_arr)
        board = torch.from_numpy(board_arr_mapped).long()
        board_orig = torch.from_numpy(board_arr).long()
        target_val = int(self.targets[idx])
        target = torch.tensor(target_val).long()

        if self.mode == "reconstruct":
            if self.mask_ratio is None:
                assert self.mask_ratio_range is not None  # noqa: S101
                low, high = self.mask_ratio_range
                ratio = torch.rand(1).mul_(high - low).add_(low).item()
            else:
                ratio = self.mask_ratio
            mask = torch.rand(board.shape) < ratio
            mask |= board_orig == target_val
            mask |= board_orig == BLANK_VALUE
        elif self.mode == "target":
            mask = (board_orig == target_val) | (board_orig == BLANK_VALUE)
        elif self.mode == "patch":
            board2d = self.boards[idx]
            r, c = np.argwhere(board2d == target_val)[0]
            mask_np = mask_target_patch(board2d, (r, c), patch_size=self.patch_size)
            mask_np |= board2d == BLANK_VALUE
            mask = torch.from_numpy(mask_np.flatten())
            # debug 輸出已移除
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        inp = board.clone()
        inp[mask] = MASK_TOKEN_ID

        orig = torch.where(mask, board, torch.full_like(board, MASK_TOKEN_ID))

        return {
            "input_vals": inp,
            "mask": mask,
            "orig_vals": orig,
            "target": target,
        }
