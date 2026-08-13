"""Pinhole camera-coordinate helpers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from utils.warp import coordinate_grid


def intrinsics_matrix(values: Sequence[Sequence[float]], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.as_tensor(values, device=device, dtype=dtype)
    if matrix.shape != (3, 3):
        raise ValueError("camera intrinsics must be a 3x3 matrix")
    if abs(float(torch.det(matrix))) < 1.0e-12:
        raise ValueError("camera intrinsics matrix is singular")
    return matrix


def interpolate_intrinsics(wide_k: torch.Tensor, tele_k: torch.Tensor, alpha: float) -> torch.Tensor:
    if wide_k.shape != (3, 3) or tele_k.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    result = (1.0 - alpha) * wide_k + alpha * tele_k
    # Focal lengths naturally evolve multiplicatively across zoom ratios.
    for index in (0, 1):
        result[index, index] = torch.exp(
            (1.0 - alpha) * torch.log(wide_k[index, index])
            + alpha * torch.log(tele_k[index, index])
        )
    return result


def remap_intrinsics(image: torch.Tensor, source_k: torch.Tensor, target_k: torch.Tensor) -> tuple:
    """Map source image to target intrinsics using p_source=K_source K_target^-1 p."""
    batch, _, height, width = image.shape
    grid = coordinate_grid(batch, height, width, image.device, image.dtype)
    homogeneous = torch.cat((grid, torch.ones_like(grid[:, :1])), dim=1).flatten(2)
    transform = source_k @ torch.linalg.inv(target_k)
    source = torch.matmul(transform.unsqueeze(0), homogeneous)
    source_xy = source[:, :2] / source[:, 2:3].clamp_min(1.0e-8)
    source_xy = source_xy.view(batch, 2, height, width)
    x, y = source_xy[:, 0], source_xy[:, 1]
    normalized_x = 2.0 * x / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * y / max(height - 1, 1) - 1.0
    sample_grid = torch.stack((normalized_x, normalized_y), dim=-1)
    mapped = F.grid_sample(image, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = ((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)).unsqueeze(1).to(image.dtype)
    return mapped * valid, valid


__all__ = ["interpolate_intrinsics", "intrinsics_matrix", "remap_intrinsics"]

