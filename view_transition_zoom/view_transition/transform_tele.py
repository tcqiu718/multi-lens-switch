"""Construct T->O flow and transform Tele into mixed-view coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from utils.warp import backward_warp, forward_warp
from view_transition.empty_fill import fill_empty_regions


@dataclass
class TeleTransformResult:
    image: torch.Tensor
    flow_t2o: torch.Tensor
    raw_flow_t2o: torch.Tensor
    splat_weight: torch.Tensor
    valid_mask: torch.Tensor
    hole_mask: torch.Tensor


def transform_tele_to_output(
    tele: torch.Tensor,
    transformed_flow: torch.Tensor,
    delta_flow: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    splat_mode: str = "bilinear",
    fill_mode: str = "background_nearest",
) -> TeleTransformResult:
    """Implement paper Eq. (7), then backward-warp Tele with the filled flow."""
    raw, weight, valid = forward_warp(transformed_flow, delta_flow, mode=splat_mode)
    background = None if foreground_mask is None else 1.0 - foreground_mask
    if background is not None:
        background, _, background_valid = forward_warp(background, delta_flow, mode=splat_mode)
        background = background * background_valid
    filled = fill_empty_regions(raw, valid, mode=fill_mode, background_mask=background)
    image, source_valid = backward_warp(tele, filled, return_valid=True)
    final_valid = source_valid
    return TeleTransformResult(image, filled, raw, weight, final_valid, 1.0 - valid)


__all__ = ["TeleTransformResult", "transform_tele_to_output"]

