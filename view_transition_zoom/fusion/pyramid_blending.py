"""True Laplacian-pyramid blending implemented in PyTorch."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def _gaussian_blur(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    horizontal = weights.view(1, 1, 1, 5).expand(image.shape[1], 1, 1, 5)
    vertical = weights.view(1, 1, 5, 1).expand(image.shape[1], 1, 5, 1)
    image = F.conv2d(F.pad(image, (2, 2, 0, 0), mode="replicate"), horizontal, groups=image.shape[1])
    return F.conv2d(F.pad(image, (0, 0, 2, 2), mode="replicate"), vertical, groups=image.shape[1])


def _downsample(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    size = (max(1, (height + 1) // 2), max(1, (width + 1) // 2))
    return F.interpolate(_gaussian_blur(image), size=size, mode="bilinear", align_corners=False)


def _upsample(image: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def gaussian_pyramid(image: torch.Tensor, levels: int) -> List[torch.Tensor]:
    if levels < 1:
        raise ValueError("pyramid levels must be at least 1")
    minimum = min(image.shape[-2:])
    if minimum < 2 ** (levels - 1):
        raise ValueError(
            "pyramid level %d is incompatible with image size %s" % (levels, tuple(image.shape[-2:]))
        )
    result = [image]
    for _ in range(1, levels):
        result.append(_downsample(result[-1]))
    return result


def laplacian_pyramid(image: torch.Tensor, levels: int) -> List[torch.Tensor]:
    gaussian = gaussian_pyramid(image, levels)
    laplacian = []
    for index in range(levels - 1):
        expanded = _upsample(gaussian[index + 1], gaussian[index].shape[-2:])
        laplacian.append(gaussian[index] - expanded)
    laplacian.append(gaussian[-1])
    return laplacian


def pyramid_blend(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    levels: int = 5,
) -> torch.Tensor:
    """Blend with semantics ``mask*first + (1-mask)*second`` at every scale."""
    if first.shape != second.shape or first.ndim != 4:
        raise ValueError("first and second must have equal [B,C,H,W] shapes")
    if mask.shape != (first.shape[0], 1, first.shape[2], first.shape[3]):
        raise ValueError("mask must have shape [B,1,H,W]")
    first_lap = laplacian_pyramid(first, levels)
    second_lap = laplacian_pyramid(second, levels)
    masks = gaussian_pyramid(mask.clamp(0.0, 1.0), levels)
    blended = [m * a + (1.0 - m) * b for a, b, m in zip(first_lap, second_lap, masks)]
    result = blended[-1]
    for index in range(levels - 2, -1, -1):
        result = _upsample(result, blended[index].shape[-2:]) + blended[index]
    return result.clamp(0.0, 1.0)


__all__ = ["gaussian_pyramid", "laplacian_pyramid", "pyramid_blend"]

