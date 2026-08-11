"""Pure-PyTorch image quality metrics with batched operation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _prepare_pair(prediction: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes must match, got {} and {}".format(
                tuple(prediction.shape), tuple(target.shape)
            )
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if prediction.is_complex() or target.is_complex():
        raise TypeError("complex tensors are not supported")
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    elif prediction.ndim != 4:
        raise ValueError("images must have shape [C,H,W] or [B,C,H,W]")
    if any(dimension <= 0 for dimension in prediction.shape):
        raise ValueError("image dimensions must be positive")

    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    elif dtype in {torch.float16, torch.bfloat16}:
        # CPU convolution does not support every low-precision dtype, and SSIM
        # statistics are substantially more stable in float32.
        dtype = torch.float32
    return prediction.to(dtype=dtype), target.to(dtype=dtype)


def _reduce(values: Tensor, reduction: str) -> Tensor:
    normalized = str(reduction).lower()
    if normalized == "none":
        return values
    if normalized == "mean":
        return values.mean()
    if normalized == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def psnr(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    eps: float = 1.0e-12,
    reduction: str = "mean",
) -> Tensor:
    """Compute peak signal-to-noise ratio independently for each batch item.

    The default reduction returns one scalar, while ``reduction='none'``
    returns ``[B]``.  MSE is lower-bounded by ``eps``, making identical images
    produce a large finite value rather than an infinity that can destabilize
    aggregate metric logging.
    """

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    prediction, target = _prepare_pair(prediction, target)
    mse = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    maximum_squared = prediction.new_tensor(float(data_range) ** 2)
    values = 10.0 * torch.log10(maximum_squared / mse.clamp_min(float(eps)))
    return _reduce(values, reduction)


def _gaussian_kernel(
    channels: int,
    window_size: int,
    sigma: float,
    reference: Tensor,
) -> Tensor:
    coordinates = torch.arange(window_size, dtype=reference.dtype, device=reference.device)
    coordinates = coordinates - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum().clamp_min(torch.finfo(reference.dtype).eps)
    kernel_2d = torch.outer(gaussian, gaussian)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def _filter(image: Tensor, kernel: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return F.conv2d(image, kernel, groups=image.shape[1])
    height, width = image.shape[-2:]
    # Reflect padding best matches local image statistics.  Replication is the
    # well-defined fallback for very small (including 1-pixel) inputs.
    mode = "reflect" if height > padding and width > padding else "replicate"
    padded = F.pad(image, (padding, padding, padding, padding), mode=mode)
    return F.conv2d(padded, kernel, groups=image.shape[1])


def ssim(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: str = "mean",
) -> Tensor:
    """Compute a numerically stable, differentiable structural similarity.

    This implementation is entirely PyTorch and supports both CHW and BCHW
    tensors.  For images smaller than ``window_size``, the largest valid odd
    window is selected automatically.  ``reduction='none'`` returns one SSIM
    score per batch item.
    """

    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if k1 <= 0 or k2 <= 0:
        raise ValueError("k1 and k2 must be positive")

    prediction, target = _prepare_pair(prediction, target)
    height, width = prediction.shape[-2:]
    effective_size = min(window_size, height, width)
    if effective_size % 2 == 0:
        effective_size -= 1
    effective_size = max(1, effective_size)

    channels = prediction.shape[1]
    kernel = _gaussian_kernel(channels, effective_size, float(sigma), prediction)
    padding = effective_size // 2

    mu_prediction = _filter(prediction, kernel, padding)
    mu_target = _filter(target, kernel, padding)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target

    variance_prediction = _filter(prediction.square(), kernel, padding) - mu_prediction_sq
    variance_target = _filter(target.square(), kernel, padding) - mu_target_sq
    covariance = _filter(prediction * target, kernel, padding) - mu_product
    # Round-off can make a true zero variance slightly negative.
    variance_prediction = variance_prediction.clamp_min(0.0)
    variance_target = variance_target.clamp_min(0.0)

    c1 = prediction.new_tensor((float(k1) * float(data_range)) ** 2)
    c2 = prediction.new_tensor((float(k2) * float(data_range)) ** 2)
    numerator = (2.0 * mu_product + c1) * (2.0 * covariance + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        variance_prediction + variance_target + c2
    )
    # C1/C2 already keep the denominator strictly positive.  Machine epsilon
    # is far too large here (for black flat patches C1*C2 is about 9e-8 in
    # float32), and clamping to eps would incorrectly make identical black
    # images score roughly 0.75 instead of 1.0.
    denominator = denominator.clamp_min(torch.finfo(prediction.dtype).tiny)
    similarity_map = (numerator / denominator).clamp(min=-1.0, max=1.0)
    values = similarity_map.flatten(start_dim=1).mean(dim=1)
    return _reduce(values, reduction)


peak_signal_noise_ratio = psnr
structural_similarity = ssim


__all__ = [
    "peak_signal_noise_ratio",
    "psnr",
    "ssim",
    "structural_similarity",
]
