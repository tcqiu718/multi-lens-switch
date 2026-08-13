"""Forward-splat Wide into mixed-view coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from utils.warp import forward_warp


@dataclass
class WideTransformResult:
    image: torch.Tensor
    weight: torch.Tensor
    valid_mask: torch.Tensor


def _offset_values(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("offset step must be positive")
    values = []
    value = float(start)
    while value <= float(stop) + step * 0.25:
        values.append(value)
        value += step
    if not values:
        raise ValueError("offset range is empty")
    return values


def transform_wide_to_output(
    wide: torch.Tensor,
    delta_flow: torch.Tensor,
    splat_mode: str = "bilinear",
    multi_warp_average: bool = True,
    offset_min: float = -0.5,
    offset_max: float = 0.5,
    offset_step: float = 0.2,
    eps: float = 1.0e-6,
) -> WideTransformResult:
    """Transform W->O using valid-weighted multi-offset forward splatting."""
    if torch.count_nonzero(delta_flow).item() == 0:
        weight = torch.ones_like(wide[:, :1])
        return WideTransformResult(wide.clone(), weight, weight.clone())
    if not multi_warp_average:
        image, weight, valid = forward_warp(wide, delta_flow, mode=splat_mode)
        return WideTransformResult(image, weight, valid)

    numerator = torch.zeros_like(wide)
    denominator = wide.new_zeros((wide.shape[0], 1, wide.shape[2], wide.shape[3]))
    values = _offset_values(offset_min, offset_max, offset_step)
    for offset_y in values:
        for offset_x in values:
            offset_flow = delta_flow.clone()
            offset_flow[:, 0] += offset_x
            offset_flow[:, 1] += offset_y
            warped, weight, _ = forward_warp(wide, offset_flow, mode=splat_mode)
            numerator += warped * weight
            denominator += weight
    valid = denominator > eps
    result = numerator / denominator.clamp_min(eps)
    result = result * valid.to(result.dtype)
    return WideTransformResult(result, denominator, valid.to(wide.dtype))


__all__ = ["WideTransformResult", "transform_wide_to_output"]
