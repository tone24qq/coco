from __future__ import annotations

import torch


def apply_dep_bias(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    target: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    N: torch.Tensor,
    dep_alpha: float,
) -> None:
    """Add row/column dependency bias to masked positions in-place."""
    device = logits.device
    B, _, V = logits.shape
    Vmax = int(N.max().item())
    mask_pos = ((tokens == 0) & (target != 0)).nonzero(as_tuple=False)
    grids = []
    for b in range(B):
        R = int(rows[b].item())
        C = int(cols[b].item())
        grids.append(tokens[b, : R * C].view(R, C))
    eps = 1e-6
    for b, lidx in mask_pos.tolist():
        R = int(rows[b].item())
        C = int(cols[b].item())
        r = lidx // C
        c = lidx % C
        grid_b = grids[b]
        row_vals = grid_b[r, :]
        col_vals = grid_b[:, c]
        vec = torch.cat([row_vals, col_vals], dim=0)
        hist = torch.bincount(vec, minlength=Vmax + 1).to(device)
        hist[0] = 0
        prior = hist.float()
        if hist.sum() > 0:
            prior = prior / hist.sum()
        else:
            prior.zero_()
        log_prior = torch.log(prior + eps) * dep_alpha
        bias = torch.full((V,), -1e9, device=device)
        length = min(int(N[b].item()) + 1, V, log_prior.numel())
        bias[:length] = log_prior[:length]
        logits[b, lidx, :] += bias
