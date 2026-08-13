"""Paper target-flow equations (2)-(5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.flow_utils import flow_magnitude, same_box_filter


@dataclass
class TargetFlowResult:
    mean_flow: torch.Tensor
    foreground_mask: torch.Tensor
    foreground_flow: torch.Tensor
    background_flow: torch.Tensor
    target_flow: torch.Tensor


def estimate_target_flow(
    flow: torch.Tensor,
    kernel_size: int = 600,
    foreground_mode: str = "magnitude",
    eps: float = 1.0e-6,
) -> TargetFlowResult:
    """Estimate a spatially smooth mixed-view target flow.

    PAPER_AMBIGUITY: the paper states that foreground/background are separated
    by comparison with local mean flow, but it does not uniquely specify the
    comparator. ``magnitude`` is the default approximation. ``component`` uses
    an any-component absolute comparison as an ablation.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    mean_flow = same_box_filter(flow, kernel_size)
    mode = foreground_mode.lower()
    if mode == "magnitude":
        mask = flow_magnitude(flow) > flow_magnitude(mean_flow)
    elif mode == "component":
        mask = torch.any(torch.abs(flow) > torch.abs(mean_flow), dim=1, keepdim=True)
    else:
        raise ValueError("foreground_mode must be magnitude or component")
    mask_f = mask.to(flow.dtype)
    inverse = 1.0 - mask_f

    foreground_den = same_box_filter(mask_f, kernel_size)
    background_den = same_box_filter(inverse, kernel_size)
    foreground = same_box_filter(flow * mask_f, kernel_size) / foreground_den.clamp_min(eps)
    background = same_box_filter(flow * inverse, kernel_size) / background_den.clamp_min(eps)

    # Empty local classes have no paper-specified value. Falling back to the
    # local mean avoids extreme zero vectors at all-foreground/background areas.
    foreground = torch.where((foreground_den > eps).expand_as(flow), foreground, mean_flow)
    background = torch.where((background_den > eps).expand_as(flow), background, mean_flow)
    target = same_box_filter((foreground + background) * 0.5, kernel_size)
    return TargetFlowResult(mean_flow, mask_f, foreground, background, target)


__all__ = ["TargetFlowResult", "estimate_target_flow"]

