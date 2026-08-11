"""Forward-backward optical-flow consistency masks."""

from typing import Tuple, Union

import torch
from torch import Tensor, nn

from .warp import warp


def _validate_pair(flow_w2t: Tensor, flow_t2w: Tensor) -> None:
    for name, value in (("flow_w2t", flow_w2t), ("flow_t2w", flow_t2w)):
        if not isinstance(value, Tensor):
            raise TypeError("{} must be a torch.Tensor".format(name))
        if value.ndim != 4 or value.shape[1] != 2:
            raise ValueError("{} must have shape [B,2,H,W]".format(name))
        if not value.is_floating_point():
            raise TypeError("{} must be floating point".format(name))
    if flow_w2t.shape != flow_t2w.shape:
        raise ValueError(
            "forward and backward flows must have the same shape, got {} and {}".format(
                tuple(flow_w2t.shape), tuple(flow_t2w.shape)
            )
        )
    if flow_w2t.device != flow_t2w.device:
        raise ValueError("forward and backward flows must be on the same device")


def compute_occlusion_mask(
    flow_w2t: Tensor,
    flow_t2w: Tensor,
    mode: str = "soft",
    threshold: float = 1.0,
    temperature: float = 0.5,
    align_corners: bool = True,
    return_error: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute a Wide-coordinate forward-backward consistency mask.

    ``flow_w2t`` maps each Wide target pixel to its Tele sample location.  The
    reverse flow is therefore first warped to Wide coordinates and consistency
    error is ``||flow_w2t + warp(flow_t2w, flow_w2t)||_2``.

    Mask semantics are project-wide: zero means reliable and one means
    occluded/unreliable.  Locations outside the Tele source bounds are always
    exactly one, independently of hard/soft mode.

    Args:
        mode: ``'hard'`` for ``error > threshold`` or ``'soft'`` for
            ``sigmoid((error-threshold)/temperature)``.
        return_error: Also return the ``[B,1,H,W]`` consistency error.
    """
    _validate_pair(flow_w2t, flow_t2w)
    if float(threshold) < 0.0:
        raise ValueError("threshold must be non-negative")
    normalized_mode = mode.lower()
    if normalized_mode not in ("hard", "soft"):
        raise ValueError("mode must be 'hard' or 'soft', got {!r}".format(mode))
    if normalized_mode == "soft" and float(temperature) <= 0.0:
        raise ValueError("temperature must be positive in soft mode")

    reverse_in_forward_dtype = flow_t2w.to(dtype=flow_w2t.dtype)
    warped_reverse, valid_mask = warp(
        reverse_in_forward_dtype,
        flow_w2t,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
        return_mask=True,
    )
    residual = flow_w2t + warped_reverse
    # torch.linalg.vector_norm has a well-defined zero gradient at a zero
    # residual, unlike a hand-written sqrt(sum(x**2)) at exactly zero.
    error = torch.linalg.vector_norm(residual, ord=2, dim=1, keepdim=True)
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))

    if normalized_mode == "hard":
        mask = (error > float(threshold)).to(dtype=flow_w2t.dtype)
    else:
        mask = torch.sigmoid((error - float(threshold)) / float(temperature))

    # valid_mask is 1 for a legal Tele sample.  This expression preserves soft
    # confidence in-bounds while making every out-of-bounds pixel exactly 1.
    valid_mask = valid_mask.to(dtype=mask.dtype)
    mask = 1.0 - (1.0 - mask) * valid_mask
    mask = mask.clamp(0.0, 1.0)
    return (mask, error) if return_error else mask


class OcclusionMask(nn.Module):
    """Configurable module wrapper for forward-backward consistency."""

    def __init__(
        self,
        mode: str = "soft",
        threshold: float = 1.0,
        temperature: float = 0.5,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        if mode.lower() not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if float(threshold) < 0.0:
            raise ValueError("threshold must be non-negative")
        if mode.lower() == "soft" and float(temperature) <= 0.0:
            raise ValueError("temperature must be positive in soft mode")
        self.mode = mode.lower()
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.align_corners = bool(align_corners)

    def forward(
        self,
        flow_w2t: Tensor,
        flow_t2w: Tensor,
        return_error: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return compute_occlusion_mask(
            flow_w2t,
            flow_t2w,
            mode=self.mode,
            threshold=self.threshold,
            temperature=self.temperature,
            align_corners=self.align_corners,
            return_error=return_error,
        )


ForwardBackwardOcclusionMask = OcclusionMask
