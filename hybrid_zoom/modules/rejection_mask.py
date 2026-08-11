"""Patch-based alignment rejection for Wide and warped Tele luminance."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .preprocessing import extract_luminance


def _to_luminance(image: Tensor, name: str) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if image.ndim != 4:
        raise ValueError("{} must have shape [B,C,H,W]".format(name))
    if not image.is_floating_point():
        raise TypeError("{} must be floating point".format(name))
    if image.shape[1] == 1:
        return image
    if image.shape[1] == 3:
        return extract_luminance(image)
    raise ValueError("{} must contain one luminance channel or three RGB channels".format(name))


def _validate_parameters(
    patch_size: int,
    stride: int,
    metric: str,
    threshold: float,
    temperature: float,
    eps: float,
) -> str:
    if isinstance(patch_size, bool) or int(patch_size) != patch_size or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if isinstance(stride, bool) or int(stride) != stride or stride <= 0:
        raise ValueError("stride must be a positive integer")
    normalized_metric = metric.lower()
    aliases = {"l1": "normalized_l1", "l2": "normalized_l2"}
    normalized_metric = aliases.get(normalized_metric, normalized_metric)
    if normalized_metric not in ("normalized_l1", "normalized_l2"):
        raise ValueError("metric must be 'normalized_l1' or 'normalized_l2'")
    if float(threshold) < 0.0:
        raise ValueError("threshold must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    return normalized_metric


def _pad_to_patch_grid(image: Tensor, patch_size: int, stride: int) -> Tuple[Tensor, int, int]:
    height, width = image.shape[-2:]
    steps_h = max(0, int(math.ceil(max(0, height - patch_size) / float(stride))))
    steps_w = max(0, int(math.ceil(max(0, width - patch_size) / float(stride))))
    padded_h = patch_size + steps_h * stride
    padded_w = patch_size + steps_w * stride
    pad_bottom = padded_h - height
    pad_right = padded_w - width
    padded = F.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")
    return padded, steps_h + 1, steps_w + 1


def _locally_normalized_patches(image: Tensor, patch_size: int, stride: int, eps: float) -> Tuple[Tensor, int, int]:
    padded, rows, columns = _pad_to_patch_grid(image, patch_size, stride)
    # [B, patch_size**2, rows*columns] because input is luminance-only.
    patches = F.unfold(padded, kernel_size=patch_size, stride=stride)
    centred = patches - patches.mean(dim=1, keepdim=True)
    variance = torch.mean(centred * centred, dim=1, keepdim=True)
    normalized = centred * torch.rsqrt(variance + float(eps))
    return normalized, rows, columns


def compute_rejection_mask(
    wide: Tensor,
    warped_tele: Tensor,
    patch_size: int = 16,
    stride: int = 8,
    metric: str = "normalized_l1",
    threshold: float = 0.25,
    temperature: float = 0.1,
    eps: float = 1e-6,
    valid_mask: Optional[Tensor] = None,
    return_score: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Estimate residual misalignment from locally normalized patches.

    Each luminance patch is independently zero-centred and divided by its RMS
    contrast.  Normalized L1 or L2 patch distance is then evaluated on an
    ``unfold`` grid and bilinearly restored to full resolution.  This rejects
    structural disagreement while being less sensitive to local gain/offset
    differences than a raw per-pixel absolute difference.

    The returned mask is ``[B,1,H,W]`` in ``[0,1]`` with the shared convention
    ``0 = reliable`` and ``1 = reject Tele``.  ``valid_mask`` can be the mask
    returned by :func:`hybrid_zoom.modules.warp.warp`; invalid pixels are forced
    to one.
    """
    normalized_metric = _validate_parameters(
        patch_size, stride, metric, threshold, temperature, eps
    )
    wide_y = _to_luminance(wide, "wide")
    tele_y = _to_luminance(warped_tele, "warped_tele")
    if wide_y.shape != tele_y.shape:
        raise ValueError(
            "wide and warped_tele luminance shapes must match, got {} and {}".format(
                tuple(wide_y.shape), tuple(tele_y.shape)
            )
        )
    if wide_y.device != tele_y.device:
        raise ValueError("wide and warped_tele must be on the same device")
    tele_y = tele_y.to(dtype=wide_y.dtype)

    wide_patches, rows, columns = _locally_normalized_patches(
        wide_y, int(patch_size), int(stride), float(eps)
    )
    tele_patches, tele_rows, tele_columns = _locally_normalized_patches(
        tele_y, int(patch_size), int(stride), float(eps)
    )
    if (rows, columns) != (tele_rows, tele_columns):
        raise RuntimeError("internal patch-grid mismatch")
    difference = wide_patches - tele_patches
    if normalized_metric == "normalized_l1":
        patch_score = torch.mean(torch.abs(difference), dim=1)
    else:
        patch_score = torch.sqrt(torch.mean(difference * difference, dim=1))
    patch_score = torch.where(
        torch.isfinite(patch_score), patch_score, torch.full_like(patch_score, float("inf"))
    )

    low_resolution_score = patch_score.reshape(wide_y.shape[0], 1, rows, columns)
    full_resolution_score = F.interpolate(
        low_resolution_score,
        size=wide_y.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    full_resolution_score = torch.where(
        torch.isfinite(full_resolution_score),
        full_resolution_score,
        torch.full_like(full_resolution_score, float("inf")),
    )

    # A calibrated sigmoid maps an exact zero difference to exactly zero while
    # retaining threshold/temperature control for meaningful disagreements.
    raw_confidence = torch.sigmoid(
        (full_resolution_score - float(threshold)) / float(temperature)
    )
    baseline = torch.sigmoid(
        full_resolution_score.new_tensor(-float(threshold) / float(temperature))
    )
    rejection = ((raw_confidence - baseline) / (1.0 - baseline)).clamp(0.0, 1.0)

    if valid_mask is not None:
        if not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a torch.Tensor or None")
        expected_shape = (wide_y.shape[0], 1, wide_y.shape[-2], wide_y.shape[-1])
        if tuple(valid_mask.shape) != expected_shape:
            raise ValueError(
                "valid_mask must have shape {}, got {}".format(
                    expected_shape, tuple(valid_mask.shape)
                )
            )
        if valid_mask.device != wide_y.device:
            raise ValueError("valid_mask and images must be on the same device")
        valid = valid_mask.to(dtype=rejection.dtype).clamp(0.0, 1.0)
        rejection = 1.0 - (1.0 - rejection) * valid

    rejection = rejection.clamp(0.0, 1.0)
    return (rejection, full_resolution_score) if return_score else rejection


class RejectionMask(nn.Module):
    """Replaceable patch-based alignment-rejection module."""

    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 8,
        metric: str = "normalized_l1",
        threshold: float = 0.25,
        temperature: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        normalized_metric = _validate_parameters(
            patch_size, stride, metric, threshold, temperature, eps
        )
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.metric = normalized_metric
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        wide: Tensor,
        warped_tele: Tensor,
        valid_mask: Optional[Tensor] = None,
        return_score: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return compute_rejection_mask(
            wide,
            warped_tele,
            patch_size=self.patch_size,
            stride=self.stride,
            metric=self.metric,
            threshold=self.threshold,
            temperature=self.temperature,
            eps=self.eps,
            valid_mask=valid_mask,
            return_score=return_score,
        )


AlignmentRejectionMask = RejectionMask
