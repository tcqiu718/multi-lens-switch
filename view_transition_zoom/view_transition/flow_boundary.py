"""Distance-to-boundary maps used by the deformation constraint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import ndimage

from utils.flow_utils import flow_spatial_gradient


@dataclass
class BoundaryResult:
    distance: torch.Tensor
    simple_distance: torch.Tensor
    motion_boundary: torch.Tensor
    gradient: torch.Tensor
    threshold: torch.Tensor


def simple_boundary_distance(flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    y = torch.arange(height, device=flow.device, dtype=flow.dtype)
    x = torch.arange(width, device=flow.device, dtype=flow.dtype)
    dy = torch.minimum(y, torch.flip(y, dims=(0,)))
    dx = torch.minimum(x, torch.flip(x, dims=(0,)))
    distance = torch.minimum(dy[:, None], dx[None, :])
    return distance[None, None].expand(batch, 1, height, width)


def _distance_inside_regions(valid: torch.Tensor, max_distance: int) -> torch.Tensor:
    """CPU reference EDT to the nearest disconnected motion boundary."""
    valid_np = valid.detach().cpu().numpy() > 0.5
    output = np.zeros(valid_np.shape, dtype=np.float32)
    for batch in range(valid_np.shape[0]):
        distance = ndimage.distance_transform_edt(valid_np[batch, 0])
        output[batch, 0] = np.minimum(distance, float(max(0, max_distance)))
    return torch.from_numpy(output).to(device=valid.device, dtype=torch.float32)


def distance_to_boundary(
    flow: torch.Tensor,
    mode: str = "flow_aware",
    gradient_threshold: Optional[float] = None,
    gradient_quantile: float = 0.90,
    max_extra_distance: int = 256,
) -> BoundaryResult:
    """Compute deformation freedom near image and motion boundaries.

    ``flow_aware`` is a documented approximation because the paper does not
    publish an exact non-connected-point distance algorithm. We threshold flow
    gradient, split space at those barriers, and compute a truncated distance
    transform inside each valid motion region. Unlike distance to the full image
    edge, this keeps internal objects bounded by their own motion boundary.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    simple = simple_boundary_distance(flow)
    gradient = flow_spatial_gradient(flow)
    if isinstance(gradient_threshold, str) and gradient_threshold.lower() in ("auto", "none", "null"):
        gradient_threshold = None
    if gradient_threshold is None:
        q = min(max(float(gradient_quantile), 0.0), 1.0)
        threshold = torch.quantile(gradient.flatten(1), q, dim=1).view(-1, 1, 1, 1)
        threshold = threshold.clamp_min(1.0e-6)
    else:
        threshold = torch.full(
            (flow.shape[0], 1, 1, 1),
            float(gradient_threshold),
            device=flow.device,
            dtype=flow.dtype,
        )
    motion_boundary = gradient > threshold
    if mode.lower() == "simple":
        distance = simple
    elif mode.lower() == "flow_aware":
        region_depth = _distance_inside_regions(~motion_boundary, max_extra_distance).to(flow.dtype)
        distance = torch.minimum(simple, region_depth)
    else:
        raise ValueError("boundary mode must be simple or flow_aware")
    return BoundaryResult(
        distance=distance,
        simple_distance=simple,
        motion_boundary=motion_boundary.to(flow.dtype),
        gradient=gradient,
        threshold=threshold,
    )


__all__ = ["BoundaryResult", "distance_to_boundary", "simple_boundary_distance"]
