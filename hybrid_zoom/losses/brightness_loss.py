"""Low-frequency brightness-consistency loss."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rgb_to_luminance(image: Tensor) -> Tensor:
    """Convert an RGB BCHW/CHW tensor to one luminance channel."""

    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] not in {1, 3}:
        raise ValueError("Expected one- or three-channel CHW/BCHW image")
    if image.shape[1] == 1:
        luminance = image
    else:
        coefficients = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        luminance = (image * coefficients).sum(dim=1, keepdim=True)
    return luminance.squeeze(0) if squeeze else luminance


def _same_pad(image: Tensor, padding: Tuple[int, int, int, int]) -> Tensor:
    left, right, top, bottom = padding
    height, width = image.shape[-2:]
    can_reflect = left < width and right < width and top < height and bottom < height
    mode = "reflect" if can_reflect else "replicate"
    return F.pad(image, padding, mode=mode)


def gaussian_blur(
    image: Tensor,
    sigma: float = 10.0,
    kernel_size: Optional[int] = None,
    truncate: float = 3.0,
) -> Tensor:
    """Apply differentiable separable Gaussian blur to CHW/BCHW tensors."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("gaussian_blur expects CHW or BCHW input")
    if kernel_size is None:
        kernel_size = 2 * int(math.ceil(truncate * sigma)) + 1
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    original_dtype = image.dtype
    working = image.float() if image.dtype in {torch.float16, torch.bfloat16} else image
    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=working.dtype)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)
    channels = working.shape[1]

    horizontal = kernel.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    vertical = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
    working = _same_pad(working, (radius, radius, 0, 0))
    working = F.conv2d(working, horizontal, groups=channels)
    working = _same_pad(working, (0, 0, radius, radius))
    working = F.conv2d(working, vertical, groups=channels)
    working = working.to(dtype=original_dtype)
    return working.squeeze(0) if squeeze else working


class BrightnessLoss(nn.Module):
    """L1 difference between heavily blurred prediction/target luminance."""

    def __init__(self, sigma: float = 10.0, kernel_size: Optional[int] = None) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.kernel_size = kernel_size

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes")
        prediction_y = rgb_to_luminance(prediction)
        target_y = rgb_to_luminance(target)
        prediction_low = gaussian_blur(prediction_y, self.sigma, self.kernel_size)
        target_low = gaussian_blur(target_y, self.sigma, self.kernel_size)
        return F.l1_loss(prediction_low, target_low)


__all__ = ["BrightnessLoss", "gaussian_blur", "rgb_to_luminance"]
