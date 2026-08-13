"""Apply the paper deformation constraint and expose continuous interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class ConstrainedFlowResult:
    transformed_flow: torch.Tensor
    delta_flow: torch.Tensor
    limit: torch.Tensor


def constrain_target_flow(
    original_flow: torch.Tensor,
    target_flow: torch.Tensor,
    distance: torch.Tensor,
    ratio: float = 0.01,
) -> ConstrainedFlowResult:
    """Component-wise clip target flow to original +/- rho * distance."""
    if original_flow.shape != target_flow.shape:
        raise ValueError("original_flow and target_flow must have equal shapes")
    if distance.shape != (original_flow.shape[0], 1, original_flow.shape[2], original_flow.shape[3]):
        raise ValueError("distance must have shape [B,1,H,W]")
    if ratio < 0:
        raise ValueError("ratio must be non-negative")
    limit = distance * float(ratio)
    transformed = torch.maximum(
        torch.minimum(target_flow, original_flow + limit),
        original_flow - limit,
    )
    return ConstrainedFlowResult(transformed, transformed - original_flow, limit)


def interpolate_transformation(
    original_flow: torch.Tensor,
    paper_delta: torch.Tensor,
    alpha: Union[float, torch.Tensor],
) -> ConstrainedFlowResult:
    """Create F_hat(z)=F_original+alpha(z)*delta_paper."""
    alpha_tensor = torch.as_tensor(alpha, device=original_flow.device, dtype=original_flow.dtype)
    while alpha_tensor.ndim < original_flow.ndim:
        alpha_tensor = alpha_tensor.unsqueeze(-1)
    delta = paper_delta * alpha_tensor
    return ConstrainedFlowResult(original_flow + delta, delta, torch.abs(delta))


__all__ = ["ConstrainedFlowResult", "constrain_target_flow", "interpolate_transformation"]

