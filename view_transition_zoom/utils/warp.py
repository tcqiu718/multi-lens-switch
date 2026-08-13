from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _check_image_flow(image: torch.Tensor, flow: torch.Tensor) -> None:
    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W]")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    if image.shape[0] != flow.shape[0] or image.shape[-2:] != flow.shape[-2:]:
        raise ValueError("image and flow batch/spatial dimensions must match")
    if image.device != flow.device:
        raise ValueError("image and flow must be on the same device")


def coordinate_grid(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=0)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def _normalize_grid(grid_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = grid_xy[:, 0]
    y = grid_xy[:, 1]
    if width > 1:
        x = 2.0 * x / float(width - 1) - 1.0
    else:
        x = torch.zeros_like(x)
    if height > 1:
        y = 2.0 * y / float(height - 1) - 1.0
    else:
        y = torch.zeros_like(y)
    return torch.stack((x, y), dim=-1)


def backward_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    return_valid: bool = False,
) -> torch.Tensor:
    """Sample ``image`` at p + flow(p), where flow is defined on output pixels.

    A constant positive dx therefore moves visible image content to the left.
    This is the canonical T->W convention used throughout this project.
    """
    _check_image_flow(image, flow)
    batch, _, height, width = image.shape
    base = coordinate_grid(batch, height, width, flow.device, flow.dtype)
    sample_xy = base + flow
    finite = torch.isfinite(sample_xy).all(dim=1, keepdim=True)
    safe_xy = torch.where(finite.expand_as(sample_xy), sample_xy, base)
    grid = _normalize_grid(safe_xy, height, width)
    warped = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    valid = (
        finite
        & (sample_xy[:, 0:1] >= 0.0)
        & (sample_xy[:, 0:1] <= float(width - 1))
        & (sample_xy[:, 1:2] >= 0.0)
        & (sample_xy[:, 1:2] <= float(height - 1))
    )
    if padding_mode == "zeros":
        warped = warped * valid.to(warped.dtype)
    if return_valid:
        return warped, valid.to(image.dtype)
    return warped


def _scatter_neighbor(
    image_flat: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    weight: torch.Tensor,
    output_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    height: int,
    width: int,
) -> None:
    finite = torch.isfinite(target_x) & torch.isfinite(target_y) & torch.isfinite(weight)
    valid = (
        finite
        & (target_x >= 0)
        & (target_x < width)
        & (target_y >= 0)
        & (target_y < height)
        & (weight > 0)
    )
    safe_x = torch.where(valid, target_x, torch.zeros_like(target_x)).long()
    safe_y = torch.where(valid, target_y, torch.zeros_like(target_y)).long()
    index = safe_y * width + safe_x
    safe_weight = torch.where(valid, weight, torch.zeros_like(weight))
    output_sum.scatter_add_(
        2,
        index.unsqueeze(1).expand(-1, image_flat.shape[1], -1),
        image_flat * safe_weight.unsqueeze(1),
    )
    weight_sum.scatter_add_(2, index.unsqueeze(1), safe_weight.unsqueeze(1))


def forward_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-splat source pixels from p to p + flow(p).

    Returns the normalized result, accumulated splat weights, and a binary valid
    mask. Multiple source pixels are accumulated and holes retain zero weight.
    A constant positive dx moves image content to the right.
    """
    _check_image_flow(image, flow)
    if mode not in ("nearest", "bilinear"):
        raise ValueError("forward_warp mode must be 'nearest' or 'bilinear'")

    batch, channels, height, width = image.shape
    base = coordinate_grid(batch, height, width, flow.device, flow.dtype)
    target = base + flow
    x = target[:, 0].reshape(batch, -1)
    y = target[:, 1].reshape(batch, -1)
    image_flat = image.reshape(batch, channels, -1)
    output_sum = image.new_zeros((batch, channels, height * width))
    weight_sum = image.new_zeros((batch, 1, height * width))

    if mode == "nearest":
        _scatter_neighbor(
            image_flat,
            torch.round(x),
            torch.round(y),
            torch.ones_like(x),
            output_sum,
            weight_sum,
            height,
            width,
        )
    else:
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = x - x0
        wy1 = y - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        _scatter_neighbor(image_flat, x0, y0, wx0 * wy0, output_sum, weight_sum, height, width)
        _scatter_neighbor(image_flat, x1, y0, wx1 * wy0, output_sum, weight_sum, height, width)
        _scatter_neighbor(image_flat, x0, y1, wx0 * wy1, output_sum, weight_sum, height, width)
        _scatter_neighbor(image_flat, x1, y1, wx1 * wy1, output_sum, weight_sum, height, width)

    valid = weight_sum > eps
    normalized = output_sum / weight_sum.clamp_min(eps)
    normalized = normalized * valid.to(normalized.dtype)
    return (
        normalized.reshape(batch, channels, height, width),
        weight_sum.reshape(batch, 1, height, width),
        valid.to(image.dtype).reshape(batch, 1, height, width),
    )

