"""Overlap and full-Wide-view multiband fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from fusion.pyramid_blending import pyramid_blend


def soften_overlap_mask(mask: torch.Tensor, width: int = 100) -> torch.Tensor:
    """Feather inward so pixels outside Tele's true FOV remain purely Wide."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("overlap mask must have shape [B,1,H,W]")
    if width <= 0:
        return (mask > 0.5).to(mask.dtype)
    output = np.zeros(tuple(mask.shape), dtype=np.float32)
    source = mask.detach().cpu().numpy() > 0.5
    for batch in range(mask.shape[0]):
        hard = source[batch, 0]
        if np.all(hard):
            output[batch, 0] = 1.0
        elif np.any(hard):
            inside_distance = ndimage.distance_transform_edt(hard)
            output[batch, 0] = np.minimum(inside_distance / float(width), 1.0)
    return torch.from_numpy(output).to(device=mask.device, dtype=mask.dtype)


@dataclass
class FusionResult:
    overlap_result: torch.Tensor
    full_result: torch.Tensor
    occlusion_mask: torch.Tensor
    overlap_mask: torch.Tensor


def full_view_fusion(
    wide_output: torch.Tensor,
    tele_output: torch.Tensor,
    wide_full: torch.Tensor,
    occlusion_mask: torch.Tensor,
    overlap_mask: torch.Tensor,
    pyramid_levels: int = 5,
    overlap_soft_width: int = 100,
) -> FusionResult:
    """Fuse O overlap, then preserve Wide outside the physical Tele overlap."""
    overlap_result = pyramid_blend(wide_output, tele_output, occlusion_mask, pyramid_levels)
    overlap_soft = soften_overlap_mask(overlap_mask, overlap_soft_width)
    full_result = pyramid_blend(overlap_result, wide_full, overlap_soft, pyramid_levels)
    return FusionResult(overlap_result, full_result, occlusion_mask, overlap_soft)


__all__ = ["FusionResult", "full_view_fusion", "soften_overlap_mask"]

