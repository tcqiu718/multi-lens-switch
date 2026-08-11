"""Dynamic reliability-mask composition and fallback-to-Wide blending."""

import math
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _normalize_smoothing(smoothing: Optional[str]) -> str:
    if smoothing is None:
        return "none"
    value = smoothing.lower()
    aliases = {"avg": "average", "mean": "average", "off": "none"}
    value = aliases.get(value, value)
    if value not in ("gaussian", "average", "none"):
        raise ValueError("smoothing must be 'gaussian', 'average', or 'none'")
    return value


def _resolve_kernel_size(smoothing: str, sigma: float, kernel_size: Optional[int]) -> int:
    if smoothing == "gaussian" and float(sigma) <= 0.0:
        raise ValueError("sigma must be positive for Gaussian smoothing")
    if kernel_size is None:
        if smoothing == "gaussian":
            result = 2 * int(math.ceil(3.0 * float(sigma))) + 1
        elif smoothing == "average":
            result = 5
        else:
            result = 1
    else:
        if isinstance(kernel_size, bool) or int(kernel_size) != kernel_size:
            raise ValueError("kernel_size must be an odd positive integer")
        result = int(kernel_size)
    if result <= 0 or result % 2 == 0:
        raise ValueError("kernel_size must be an odd positive integer")
    return result


def _smooth_mask(mask: Tensor, smoothing: str, sigma: float, kernel_size: int) -> Tensor:
    if smoothing == "none" or kernel_size == 1:
        return mask
    radius = kernel_size // 2
    padded = F.pad(mask, (radius, radius, radius, radius), mode="replicate")
    if smoothing == "average":
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    coordinates = torch.arange(kernel_size, device=mask.device, dtype=mask.dtype) - radius
    kernel_1d = torch.exp(-(coordinates * coordinates) / (2.0 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = mask.shape[1]
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(channels, 1, -1, -1)
    return F.conv2d(padded, weight, groups=channels)


def compute_blend_mask(
    masks: Mapping[str, Optional[Tensor]],
    smoothing: Optional[str] = "gaussian",
    sigma: float = 1.5,
    kernel_size: Optional[int] = None,
) -> Tensor:
    """Dynamically combine any available unreliability masks.

    Every non-``None`` value in ``masks`` participates in
    ``clamp(1 - sum(unreliability), 0, 1)``.  This directly supports future
    entries such as ``flow_uncertainty`` and ``defocus`` without changing the
    implementation.  All input masks use ``0 = reliable, 1 = unreliable``;
    the returned blend mask uses the complementary convention ``1 = use
    fusion/Tele, 0 = fall back to Wide``.
    """
    if not isinstance(masks, Mapping):
        raise TypeError("masks must be a mapping from names to tensors or None")
    active = [(name, value) for name, value in masks.items() if value is not None]
    if not active:
        raise ValueError("at least one non-None mask is required to infer shape and device")

    first_name, first_mask = active[0]
    if not isinstance(first_mask, Tensor):
        raise TypeError("mask {!r} must be a torch.Tensor or None".format(first_name))
    if first_mask.ndim != 4 or first_mask.shape[1] != 1:
        raise ValueError("mask {!r} must have shape [B,1,H,W]".format(first_name))
    if not first_mask.is_floating_point():
        raise TypeError("mask {!r} must be floating point".format(first_name))
    if first_mask.shape[0] <= 0 or first_mask.shape[-2] <= 0 or first_mask.shape[-1] <= 0:
        raise ValueError("mask dimensions must be non-empty")

    unreliability = torch.zeros_like(first_mask)
    for name, value in active:
        if not isinstance(value, Tensor):
            raise TypeError("mask {!r} must be a torch.Tensor or None".format(name))
        if value.shape != first_mask.shape:
            raise ValueError(
                "all masks must share shape {}; {!r} has {}".format(
                    tuple(first_mask.shape), name, tuple(value.shape)
                )
            )
        if value.device != first_mask.device:
            raise ValueError("all masks must be on the same device")
        if not value.is_floating_point():
            raise TypeError("mask {!r} must be floating point".format(name))
        sanitized = torch.where(
            torch.isfinite(value),
            value.to(dtype=first_mask.dtype).clamp(0.0, 1.0),
            torch.ones_like(value, dtype=first_mask.dtype),
        )
        unreliability = unreliability + sanitized

    blend_mask = (1.0 - unreliability).clamp(0.0, 1.0)
    normalized_smoothing = _normalize_smoothing(smoothing)
    resolved_kernel = _resolve_kernel_size(normalized_smoothing, sigma, kernel_size)
    blend_mask = _smooth_mask(
        blend_mask, normalized_smoothing, float(sigma), resolved_kernel
    )
    return blend_mask.clamp(0.0, 1.0)


def adaptive_blend(
    fusion: Tensor,
    wide: Tensor,
    masks: Mapping[str, Optional[Tensor]],
    smoothing: Optional[str] = "gaussian",
    sigma: float = 1.5,
    kernel_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Blend fusion with Wide and return ``(final, blend_mask)``.

    ``fusion`` and ``wide`` may contain one luminance channel or multiple image
    channels; the single-channel blend mask broadcasts over their channels.
    """
    for name, value in (("fusion", fusion), ("wide", wide)):
        if not isinstance(value, Tensor):
            raise TypeError("{} must be a torch.Tensor".format(name))
        if value.ndim != 4:
            raise ValueError("{} must have shape [B,C,H,W]".format(name))
        if not value.is_floating_point():
            raise TypeError("{} must be floating point".format(name))
    if fusion.shape != wide.shape:
        raise ValueError("fusion and wide must have identical shapes")
    if fusion.device != wide.device:
        raise ValueError("fusion and wide must be on the same device")

    blend_mask = compute_blend_mask(
        masks, smoothing=smoothing, sigma=sigma, kernel_size=kernel_size
    )
    expected = (fusion.shape[0], 1, fusion.shape[-2], fusion.shape[-1])
    if tuple(blend_mask.shape) != expected:
        raise ValueError(
            "mask shape {} is incompatible with image shape {}".format(
                tuple(blend_mask.shape), tuple(fusion.shape)
            )
        )
    if blend_mask.device != fusion.device:
        raise ValueError("masks and images must be on the same device")
    blend_mask = blend_mask.to(dtype=fusion.dtype)
    wide_in_fusion_dtype = wide.to(dtype=fusion.dtype)
    final = blend_mask * fusion + (1.0 - blend_mask) * wide_in_fusion_dtype
    return final, blend_mask


class AdaptiveBlending(nn.Module):
    """Module form of adaptive fallback blending."""

    def __init__(
        self,
        smoothing: Optional[str] = "gaussian",
        sigma: float = 1.5,
        kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        normalized_smoothing = _normalize_smoothing(smoothing)
        resolved_kernel = _resolve_kernel_size(normalized_smoothing, sigma, kernel_size)
        self.smoothing = normalized_smoothing
        self.sigma = float(sigma)
        self.kernel_size = resolved_kernel

    def compute_mask(self, masks: Mapping[str, Optional[Tensor]]) -> Tensor:
        """Return the smoothed fusion-usage mask without applying it."""
        return compute_blend_mask(
            masks,
            smoothing=self.smoothing,
            sigma=self.sigma,
            kernel_size=self.kernel_size,
        )

    def forward(
        self,
        fusion: Tensor,
        wide: Tensor,
        masks: Mapping[str, Optional[Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        return adaptive_blend(
            fusion,
            wide,
            masks,
            smoothing=self.smoothing,
            sigma=self.sigma,
            kernel_size=self.kernel_size,
        )


AdaptiveBlender = AdaptiveBlending

