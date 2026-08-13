"""FOV mapping, deliberately separate from viewpoint transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from utils.camera_geometry import intrinsics_matrix, remap_intrinsics


@dataclass
class FovResult:
    image: torch.Tensor
    valid_mask: torch.Tensor
    crop_scale: float
    crop_box: tuple


def center_crop_fov(image: torch.Tensor, target_zoom: float, source_zoom: float = 1.0) -> FovResult:
    """Map an image captured at source_zoom into target_zoom coordinates.

    For target/source >= 1 this center-crops then resizes. For a wider target
    FOV, it shrinks the source into the center and reports invalid surroundings.
    """
    if target_zoom <= 0 or source_zoom <= 0:
        raise ValueError("zoom ratios must be positive")
    batch, _, height, width = image.shape
    ratio = float(target_zoom) / float(source_zoom)
    if abs(ratio - 1.0) < 1.0e-8:
        return FovResult(image.clone(), torch.ones_like(image[:, :1]), ratio, (0, 0, width, height))
    if ratio > 1.0:
        crop_h = max(1, int(round(height / ratio)))
        crop_w = max(1, int(round(width / ratio)))
        y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
        cropped = image[..., y0 : y0 + crop_h, x0 : x0 + crop_w]
        mapped = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
        return FovResult(mapped, torch.ones_like(image[:, :1]), ratio, (x0, y0, x0 + crop_w, y0 + crop_h))

    mapped_h = max(1, int(round(height * ratio)))
    mapped_w = max(1, int(round(width * ratio)))
    resized = F.interpolate(image, size=(mapped_h, mapped_w), mode="bilinear", align_corners=False)
    output = torch.zeros_like(image)
    valid = torch.zeros_like(image[:, :1])
    y0, x0 = (height - mapped_h) // 2, (width - mapped_w) // 2
    output[..., y0 : y0 + mapped_h, x0 : x0 + mapped_w] = resized
    valid[..., y0 : y0 + mapped_h, x0 : x0 + mapped_w] = 1.0
    return FovResult(output, valid, ratio, (x0, y0, x0 + mapped_w, y0 + mapped_h))


def intrinsics_fov(
    image: torch.Tensor,
    source_intrinsics: Sequence[Sequence[float]],
    target_intrinsics: Sequence[Sequence[float]],
) -> FovResult:
    source_k = intrinsics_matrix(source_intrinsics, image.device, image.dtype)
    target_k = intrinsics_matrix(target_intrinsics, image.device, image.dtype)
    mapped, valid = remap_intrinsics(image, source_k, target_k)
    scale = float(target_k[0, 0] / source_k[0, 0])
    return FovResult(mapped, valid, scale, (0, 0, image.shape[-1], image.shape[-2]))


__all__ = ["FovResult", "center_crop_fov", "intrinsics_fov"]

