"""Backward optical-flow warping in the Wide target coordinate system."""

from typing import Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_flow(flow: Tensor) -> None:
    if not isinstance(flow, Tensor):
        raise TypeError("flow must be a torch.Tensor")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W], got {}".format(tuple(flow.shape)))
    if not flow.is_floating_point():
        raise TypeError("flow must be floating point")
    if flow.shape[-2] <= 0 or flow.shape[-1] <= 0:
        raise ValueError("flow spatial dimensions must be non-empty")


def _validate_size(size: Sequence[int]) -> Tuple[int, int]:
    if len(size) != 2:
        raise ValueError("new_size must be (height, width)")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("new_size values must be positive, got {}".format((height, width)))
    return height, width


def flow_to_sampling_grid(flow: Tensor, align_corners: bool = True) -> Tuple[Tensor, Tensor]:
    """Convert pixel displacement flow to a normalized sampling grid.

    The project defines ``flow_w2t`` at every Wide (target) pixel ``p`` as
    ``p_tele = p_wide + flow_w2t(p_wide)``.  Consequently this function adds
    flow to a Wide pixel grid and produces the *backward* sampling grid used to
    pull pixels from Tele into Wide coordinates.

    Args:
        flow: ``[B,2,H,W]`` tensor, with ``flow[:,0]=dx`` and
            ``flow[:,1]=dy`` measured in pixels.
        align_corners: Passed unchanged to :func:`torch.nn.functional.grid_sample`.
            Both ``True`` and ``False`` are handled with their corresponding
            pixel-to-normalized-coordinate formula.

    Returns:
        ``(grid, valid_mask)`` where grid is ``[B,H,W,2]`` in ``[-1,1]``
        coordinates and valid_mask is boolean ``[B,1,H,W]``.  The mask uses
        the physical source-pixel bounds ``0..W-1`` and ``0..H-1``.
    """
    _validate_flow(flow)
    batch, _, height, width = flow.shape
    y_coords = torch.arange(height, device=flow.device, dtype=flow.dtype)
    x_coords = torch.arange(width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    source_x = grid_x.unsqueeze(0) + flow[:, 0]
    source_y = grid_y.unsqueeze(0) + flow[:, 1]

    finite = torch.isfinite(source_x) & torch.isfinite(source_y)
    valid = (
        finite
        & (source_x >= 0.0)
        & (source_x <= float(width - 1))
        & (source_y >= 0.0)
        & (source_y <= float(height - 1))
    )

    if align_corners:
        normalized_x = torch.zeros_like(source_x) if width == 1 else 2.0 * source_x / (width - 1) - 1.0
        normalized_y = torch.zeros_like(source_y) if height == 1 else 2.0 * source_y / (height - 1) - 1.0
    else:
        normalized_x = 2.0 * (source_x + 0.5) / width - 1.0
        normalized_y = 2.0 * (source_y + 0.5) / height - 1.0

    # Never pass NaN/Inf coordinates into grid_sample.  Invalid locations are
    # deliberately placed out of bounds and are already marked by valid_mask.
    outside_x = torch.full_like(normalized_x, 2.0)
    outside_y = torch.full_like(normalized_y, 2.0)
    normalized_x = torch.where(finite, normalized_x, outside_x)
    normalized_y = torch.where(finite, normalized_y, outside_y)
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    if grid.shape != (batch, height, width, 2):
        raise RuntimeError("internal sampling-grid shape error")
    return grid, valid.unsqueeze(1)


def warp(
    image: Tensor,
    flow: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    return_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Backward-warp ``image`` using pixel-space ``flow``.

    For the canonical call ``warp(tele, flow_w2t)``, output pixel ``(x,y)``
    samples Tele at ``(x + dx, y + dy)``.  A positive constant ``dx`` therefore
    shifts visible source content to the *left* in the returned image.  This is
    sampling flow, not forward splatting flow.

    Args:
        image: Source image/features with shape ``[B,C,H,W]``.
        flow: Target-to-source flow ``[B,2,H,W]`` in source pixels.
        return_mask: Return a same-dtype ``[B,1,H,W]`` mask containing one for
            geometrically valid source coordinates and zero otherwise.
    """
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 4:
        raise ValueError("image must have shape [B,C,H,W], got {}".format(tuple(image.shape)))
    if not image.is_floating_point():
        raise TypeError("grid_sample requires a floating-point image")
    if image.shape[1] <= 0 or image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError("image dimensions must be non-empty")
    _validate_flow(flow)
    if image.shape[0] != flow.shape[0]:
        raise ValueError("image and flow batch sizes must match")
    if image.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            "image and flow spatial sizes must match; resize flow with resize_flow first"
        )
    if image.device != flow.device:
        raise ValueError("image and flow must be on the same device")
    if mode not in ("bilinear", "nearest", "bicubic"):
        raise ValueError("unsupported grid_sample mode {!r}".format(mode))
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("unsupported padding_mode {!r}".format(padding_mode))

    # grid_sample requires the grid and input to use the same floating dtype.
    sampling_grid, valid = flow_to_sampling_grid(flow.to(dtype=image.dtype), align_corners)
    warped = F.grid_sample(
        image,
        sampling_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if return_mask:
        return warped, valid.to(dtype=image.dtype)
    return warped


def resize_flow(
    flow: Tensor,
    new_size: Sequence[int],
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Tensor:
    """Resize flow and correctly rescale its dx/dy magnitudes.

    Resizing from ``(Hf,Wf)`` to ``(H,W)`` applies ``dx *= W/Wf`` and
    ``dy *= H/Hf`` after interpolation.  Merely interpolating a flow tensor is
    incorrect because its values are measured in pixels.
    """
    _validate_flow(flow)
    new_height, new_width = _validate_size(new_size)
    old_height, old_width = flow.shape[-2:]
    if (new_height, new_width) == (old_height, old_width):
        return flow.clone()
    if mode not in ("bilinear", "nearest", "bicubic"):
        raise ValueError("unsupported interpolation mode {!r}".format(mode))
    kwargs = {"size": (new_height, new_width), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = align_corners
    resized = F.interpolate(flow, **kwargs)
    resized_x = resized[:, 0:1] * (float(new_width) / old_width)
    resized_y = resized[:, 1:2] * (float(new_height) / old_height)
    return torch.cat((resized_x, resized_y), dim=1)


class BackwardWarp(nn.Module):
    """Small module wrapper useful inside larger ``nn.Module`` pipelines."""

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)

    def forward(
        self, image: Tensor, flow: Tensor, return_mask: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return warp(
            image,
            flow,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            return_mask=return_mask,
        )


flow_warp = warp

