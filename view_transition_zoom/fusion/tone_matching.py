"""Regional histogram and temporally stable local-affine tone matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from utils.flow_utils import same_box_filter


def _histogram_match_channel(
    source: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    selected_source = source[valid]
    selected_reference = reference[valid]
    if selected_source.numel() < 8 or selected_reference.numel() < 8:
        return source
    source_hist = torch.histc(selected_source.float(), bins=bins, min=0.0, max=1.0)
    reference_hist = torch.histc(selected_reference.float(), bins=bins, min=0.0, max=1.0)
    source_cdf = torch.cumsum(source_hist, dim=0)
    reference_cdf = torch.cumsum(reference_hist, dim=0)
    source_cdf = source_cdf / source_cdf[-1].clamp_min(1.0)
    reference_cdf = reference_cdf / reference_cdf[-1].clamp_min(1.0)
    mapping = torch.searchsorted(reference_cdf, source_cdf, right=False).clamp(0, bins - 1)
    lookup = mapping.to(source.dtype) / float(bins - 1)
    index = torch.round(source.clamp(0.0, 1.0) * float(bins - 1)).long()
    return lookup[index]


def global_histogram_match(
    source: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    bins: int = 256,
) -> torch.Tensor:
    if source.shape != reference.shape or source.ndim != 4:
        raise ValueError("source and reference must have equal [B,C,H,W] shapes")
    valid = torch.ones_like(source[:, :1], dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.shape != source[:, :1].shape:
            raise ValueError("valid_mask must have shape [B,1,H,W]")
        valid = valid_mask > 0.5
    result = source.clone()
    for batch in range(source.shape[0]):
        for channel in range(source.shape[1]):
            result[batch, channel] = _histogram_match_channel(
                source[batch, channel], reference[batch, channel], valid[batch, 0], bins
            )
    return result.clamp(0.0, 1.0)


def _block_starts(length: int, block_size: int, stride: int) -> list:
    if block_size >= length:
        return [0]
    starts = list(range(0, length - block_size + 1, stride))
    final = length - block_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def regional_histogram_match(
    source: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    block_size: int = 200,
    stride: int = 30,
    bins: int = 256,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Paper RHE approximation with overlapping raised-Hann accumulation.

    PAPER_AMBIGUITY: block size/stride are specified by the paper; overlap
    weighting is not. A raised Hann window avoids block seams and zero-weight
    image borders.
    """
    if source.shape != reference.shape or source.ndim != 4:
        raise ValueError("source and reference must have equal [B,C,H,W] shapes")
    if block_size <= 0 or stride <= 0:
        raise ValueError("block_size and stride must be positive")
    height, width = source.shape[-2:]
    block_h, block_w = min(block_size, height), min(block_size, width)
    starts_y = _block_starts(height, block_h, stride)
    starts_x = _block_starts(width, block_w, stride)
    window_y = torch.hann_window(block_h, periodic=False, device=source.device, dtype=source.dtype).clamp_min(0.05)
    window_x = torch.hann_window(block_w, periodic=False, device=source.device, dtype=source.dtype).clamp_min(0.05)
    window = (window_y[:, None] * window_x[None, :])[None, None]
    accumulator = torch.zeros_like(source)
    weights = torch.zeros_like(source[:, :1])
    valid = torch.ones_like(source[:, :1]) if valid_mask is None else valid_mask
    if valid.shape != source[:, :1].shape:
        raise ValueError("valid_mask must have shape [B,1,H,W]")

    for y in starts_y:
        for x in starts_x:
            ys, xs = slice(y, y + block_h), slice(x, x + block_w)
            matched = global_histogram_match(
                source[..., ys, xs], reference[..., ys, xs], valid[..., ys, xs], bins=bins
            )
            accumulator[..., ys, xs] += matched * window
            weights[..., ys, xs] += window
    return (accumulator / weights.clamp_min(eps)).clamp(0.0, 1.0)


def local_affine_parameters(
    source: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    window: int = 61,
    eps: float = 1.0e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if source.shape != reference.shape:
        raise ValueError("source and reference shapes must match")
    valid = torch.ones_like(source[:, :1]) if valid_mask is None else valid_mask
    if valid.shape != source[:, :1].shape:
        raise ValueError("valid_mask must have shape [B,1,H,W]")
    denominator = same_box_filter(valid, window).clamp_min(eps)

    def moments(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = same_box_filter(image * valid, window) / denominator
        second = same_box_filter(image * image * valid, window) / denominator
        variance = (second - mean * mean).clamp_min(eps)
        return mean, variance

    source_mean, source_var = moments(source)
    reference_mean, reference_var = moments(reference)
    gain = torch.sqrt(reference_var / source_var).clamp(0.5, 2.0)
    bias = reference_mean - gain * source_mean
    return gain, bias


@dataclass
class ToneResult:
    image: torch.Tensor
    gain: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None


class ToneMatcher:
    """Stateful tone matcher; temporal state is used only by affine mode."""

    def __init__(
        self,
        mode: str = "paper_rhe",
        block_size: int = 200,
        stride: int = 30,
        histogram_bins: int = 256,
        local_window: int = 61,
        ema: float = 0.85,
    ) -> None:
        self.mode = mode
        self.block_size = block_size
        self.stride = stride
        self.histogram_bins = histogram_bins
        self.local_window = local_window
        self.ema = float(ema)
        self.previous_gain = None
        self.previous_bias = None

    def reset(self) -> None:
        self.previous_gain = None
        self.previous_bias = None

    def __call__(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> ToneResult:
        mode = self.mode.lower()
        if mode == "none":
            return ToneResult(source)
        if mode == "global":
            return ToneResult(global_histogram_match(source, reference, valid_mask, self.histogram_bins))
        if mode == "paper_rhe":
            return ToneResult(
                regional_histogram_match(
                    source,
                    reference,
                    valid_mask,
                    self.block_size,
                    self.stride,
                    self.histogram_bins,
                )
            )
        if mode not in ("local_affine", "local_affine_temporal"):
            raise ValueError("Unknown tone mode: %s" % self.mode)
        gain, bias = local_affine_parameters(source, reference, valid_mask, self.local_window)
        if mode == "local_affine_temporal" and self.previous_gain is not None:
            if self.previous_gain.shape == gain.shape:
                gain = self.ema * self.previous_gain + (1.0 - self.ema) * gain
                bias = self.ema * self.previous_bias + (1.0 - self.ema) * bias
        if mode == "local_affine_temporal":
            self.previous_gain = gain.detach()
            self.previous_bias = bias.detach()
        return ToneResult((gain * source + bias).clamp(0.0, 1.0), gain, bias)


__all__ = [
    "ToneMatcher",
    "ToneResult",
    "global_histogram_match",
    "local_affine_parameters",
    "regional_histogram_match",
]

