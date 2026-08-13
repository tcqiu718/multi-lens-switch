"""Fill holes created by forward-splatting transformed flow fields."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _nearest_fill_cpu(values: torch.Tensor, seed_mask: torch.Tensor, holes: torch.Tensor) -> torch.Tensor:
    output = values.detach().cpu().numpy().copy()
    seeds = seed_mask.detach().cpu().numpy() > 0.5
    holes_np = holes.detach().cpu().numpy() > 0.5
    for batch in range(output.shape[0]):
        valid_2d = seeds[batch, 0]
        target_2d = holes_np[batch, 0]
        if not np.any(target_2d) or not np.any(valid_2d):
            continue
        _, indices = ndimage.distance_transform_edt(~valid_2d, return_indices=True)
        yy, xx = indices[0], indices[1]
        for channel in range(output.shape[1]):
            nearest = output[batch, channel, yy, xx]
            output[batch, channel, target_2d] = nearest[target_2d]
    return torch.from_numpy(output).to(device=values.device, dtype=values.dtype)


def _iterative_fill(values: torch.Tensor, valid: torch.Tensor, iterations: int) -> torch.Tensor:
    output = values.clone()
    known = valid.clone()
    channels = values.shape[1]
    kernel = torch.ones((channels, 1, 3, 3), device=values.device, dtype=values.dtype)
    for _ in range(max(1, iterations)):
        if torch.all(known > 0.5):
            break
        sums = F.conv2d(output * known, kernel, padding=1, groups=channels)
        counts = F.conv2d(known.expand(-1, channels, -1, -1), kernel, padding=1, groups=channels)
        can_fill = (known < 0.5) & (counts[:, :1] > 0)
        proposed = sums / counts.clamp_min(1.0)
        output = torch.where(can_fill.expand_as(output), proposed, output)
        known = torch.where(can_fill, torch.ones_like(known), known)
    return output


def fill_empty_regions(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "background_nearest",
    background_mask: Optional[torch.Tensor] = None,
    iterations: Optional[int] = None,
) -> torch.Tensor:
    """Fill forward-warp holes, preferring nearby valid background samples.

    PAPER_AMBIGUITY: the paper describes background-prioritized filling but does
    not specify its interpolation algorithm. This CPU reference uses Euclidean
    nearest valid background, followed by nearest-any-valid fallback.
    """
    if valid_mask.shape != (values.shape[0], 1, values.shape[2], values.shape[3]):
        raise ValueError("valid_mask must have shape [B,1,H,W]")
    holes = valid_mask < 0.5
    if not torch.any(holes):
        return values
    no_seed = ~torch.any(valid_mask.flatten(1) > 0.5, dim=1)
    if torch.any(no_seed):
        indices = torch.nonzero(no_seed, as_tuple=False).flatten().tolist()
        raise ValueError("Cannot fill forward-warp holes: no valid splat seeds in batches %s" % indices)
    normalized = mode.lower()
    if normalized == "iterative":
        count = iterations or max(values.shape[-2:])
        return _iterative_fill(values, valid_mask, count)
    if normalized not in ("nearest", "background_nearest"):
        raise ValueError("fill mode must be background_nearest, nearest, or iterative")

    filled = values
    if normalized == "background_nearest" and background_mask is not None:
        if background_mask.shape != valid_mask.shape:
            raise ValueError("background_mask must match valid_mask")
        seeds = valid_mask * (background_mask > 0.5).to(valid_mask.dtype)
        has_background = torch.any(seeds.flatten(1) > 0.5, dim=1).view(-1, 1, 1, 1)
        background_holes = holes & has_background
        filled = _nearest_fill_cpu(filled, seeds, background_holes)
        holes = holes & (~has_background)
    filled = _nearest_fill_cpu(filled, valid_mask, holes)
    return filled


__all__ = ["fill_empty_regions"]
