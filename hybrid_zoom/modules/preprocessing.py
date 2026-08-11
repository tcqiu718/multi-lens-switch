"""Tensor-only image preprocessing for the hybrid-zoom pipeline.

All images in this project are RGB tensors.  OpenCV/BGR conventions are never
used in this module.  Functions accept either ``[C, H, W]`` or
``[B, C, H, W]`` tensors and preserve the input rank.
"""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SizeLike = Union[int, Sequence[int]]


def _parse_size(size: SizeLike, name: str = "size") -> Tuple[int, int]:
    """Convert an integer or ``(height, width)`` sequence to a size tuple."""
    if isinstance(size, bool):
        raise TypeError("{} must be an int or a two-element sequence".format(name))
    if isinstance(size, int):
        height, width = size, size
    else:
        if len(size) != 2:
            raise ValueError("{} must contain exactly (height, width)".format(name))
        height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("{} values must be positive, got {}".format(name, (height, width)))
    return height, width


def _validate_image(image: Tensor, channels: Optional[int] = None) -> None:
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim not in (3, 4):
        raise ValueError("image must have shape [C,H,W] or [B,C,H,W], got {}".format(tuple(image.shape)))
    if channels is not None and image.shape[-3] != channels:
        raise ValueError("expected {} channels, got {}".format(channels, image.shape[-3]))
    if image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError("image spatial dimensions must be non-empty")


def _require_float(image: Tensor) -> None:
    if not image.is_floating_point():
        raise TypeError("color conversion requires a floating-point tensor")


def _as_batched(image: Tensor) -> Tuple[Tensor, bool]:
    return (image.unsqueeze(0), True) if image.ndim == 3 else (image, False)


def _restore_rank(image: Tensor, was_unbatched: bool) -> Tensor:
    return image.squeeze(0) if was_unbatched else image


def _channel_transform(image: Tensor, matrix: Sequence[Sequence[float]]) -> Tensor:
    transform = image.new_tensor(matrix)
    return torch.einsum("ij,...jhw->...ihw", transform, image)


def rgb_to_ycbcr(rgb: Tensor, clamp: bool = False) -> Tensor:
    """Convert full-range RGB in ``[0, 1]`` to full-range JPEG YCbCr.

    ``Cb`` and ``Cr`` are centred at 0.5.  Clamping is disabled by default so
    that ``ycbcr_to_rgb(rgb_to_ycbcr(x))`` remains numerically reversible.
    """
    _validate_image(rgb, channels=3)
    _require_float(rgb)
    result = _channel_transform(
        rgb,
        (
            (0.299, 0.587, 0.114),
            (-0.16873589164785552, -0.3312641083521445, 0.5),
            (0.5, -0.4186875891583452, -0.08131241084165478),
        ),
    )
    offset_shape = [1] * (rgb.ndim - 3) + [3, 1, 1]
    offset = rgb.new_tensor((0.0, 0.5, 0.5)).view(offset_shape)
    result = result + offset
    return result.clamp(0.0, 1.0) if clamp else result


def ycbcr_to_rgb(ycbcr: Tensor, clamp: bool = False) -> Tensor:
    """Convert full-range JPEG YCbCr to RGB.

    Args:
        ycbcr: Tensor shaped ``[3,H,W]`` or ``[B,3,H,W]``.
        clamp: If true, clamp reconstructed RGB to ``[0, 1]``.
    """
    _validate_image(ycbcr, channels=3)
    _require_float(ycbcr)
    y = ycbcr.select(-3, 0)
    cb = ycbcr.select(-3, 1) - 0.5
    cr = ycbcr.select(-3, 2) - 0.5
    red = y + 1.402 * cr
    blue = y + 1.772 * cb
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    result = torch.stack((red, green, blue), dim=-3)
    return result.clamp(0.0, 1.0) if clamp else result


def rgb_to_yuv(rgb: Tensor, clamp: bool = False) -> Tensor:
    """Convert RGB to analogue BT.601 YUV (U and V are centred at zero).

    ``clamp=True`` clamps Y to ``[0,1]``, U to ``[-0.436,0.436]`` and V to
    ``[-0.615,0.615]``.  It is normally best to leave clamping disabled.
    """
    _validate_image(rgb, channels=3)
    _require_float(rgb)
    result = _channel_transform(
        rgb,
        (
            (0.299, 0.587, 0.114),
            (-0.147108, -0.288804, 0.435912),
            (0.614777, -0.514799, -0.099978),
        ),
    )
    if not clamp:
        return result
    y = result.select(-3, 0).clamp(0.0, 1.0)
    u = result.select(-3, 1).clamp(-0.436, 0.436)
    v = result.select(-3, 2).clamp(-0.615, 0.615)
    return torch.stack((y, u, v), dim=-3)


def yuv_to_rgb(yuv: Tensor, clamp: bool = False) -> Tensor:
    """Convert analogue BT.601 YUV back to RGB."""
    _validate_image(yuv, channels=3)
    _require_float(yuv)
    y = yuv.select(-3, 0)
    u = yuv.select(-3, 1)
    v = yuv.select(-3, 2)
    red = y + v / 0.877
    blue = y + u / 0.492
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    result = torch.stack((red, green, blue), dim=-3)
    return result.clamp(0.0, 1.0) if clamp else result


def extract_luminance(rgb: Tensor) -> Tensor:
    """Return BT.601 luminance Y with shape ``[..., 1, H, W]``."""
    _validate_image(rgb, channels=3)
    _require_float(rgb)
    red = rgb.select(-3, 0)
    green = rgb.select(-3, 1)
    blue = rgb.select(-3, 2)
    return (0.299 * red + 0.587 * green + 0.114 * blue).unsqueeze(-3)


def replace_luminance(reference_rgb: Tensor, luminance: Tensor, clamp: bool = True) -> Tensor:
    """Combine a new Y channel with the reference RGB image's Cb/Cr channels."""
    _validate_image(reference_rgb, channels=3)
    _validate_image(luminance, channels=1)
    _require_float(reference_rgb)
    _require_float(luminance)
    expected = list(reference_rgb.shape)
    expected[-3] = 1
    if tuple(luminance.shape) != tuple(expected):
        raise ValueError(
            "luminance shape {} does not match reference shape {}".format(
                tuple(luminance.shape), tuple(expected)
            )
        )
    if luminance.device != reference_rgb.device:
        raise ValueError("luminance and reference_rgb must be on the same device")
    chroma = rgb_to_ycbcr(reference_rgb).narrow(-3, 1, 2)
    ycbcr = torch.cat((luminance.to(dtype=reference_rgb.dtype), chroma), dim=-3)
    return ycbcr_to_rgb(ycbcr, clamp=clamp)


def center_crop(image: Tensor, size: SizeLike) -> Tensor:
    """Take a deterministic centre crop without resizing."""
    _validate_image(image)
    crop_h, crop_w = _parse_size(size)
    height, width = image.shape[-2:]
    if crop_h > height or crop_w > width:
        raise ValueError(
            "crop {} exceeds input spatial size {}".format((crop_h, crop_w), (height, width))
        )
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[..., top : top + crop_h, left : left + crop_w]


def resize_image(
    image: Tensor,
    size: SizeLike,
    mode: str = "bilinear",
    align_corners: Optional[bool] = False,
) -> Tensor:
    """Resize a CHW or BCHW image to ``(height, width)``."""
    _validate_image(image)
    if not image.is_floating_point():
        raise TypeError("resize_image requires a floating-point tensor")
    output_size = _parse_size(size)
    batched, was_unbatched = _as_batched(image)
    corner_modes = ("linear", "bilinear", "bicubic", "trilinear")
    kwargs = {"size": output_size, "mode": mode}
    if mode in corner_modes:
        kwargs["align_corners"] = align_corners
    resized = F.interpolate(batched, **kwargs)
    return _restore_rank(resized, was_unbatched)


def resize_with_aspect_ratio(
    image: Tensor,
    size: SizeLike,
    mode: str = "letterbox",
    interpolation: str = "bilinear",
    fill_value: float = 0.0,
    align_corners: Optional[bool] = False,
) -> Tensor:
    """Resize while preserving aspect ratio.

    ``mode='letterbox'`` fits the whole image and pads it to the requested
    shape.  ``mode='crop'`` fills the requested shape and centre-crops excess.
    The returned spatial size is always exactly ``size``.
    """
    _validate_image(image)
    target_h, target_w = _parse_size(size)
    source_h, source_w = image.shape[-2:]
    normalized_mode = mode.lower()
    if normalized_mode in ("letterbox", "fit", "pad"):
        scale = min(float(target_h) / source_h, float(target_w) / source_w)
        new_h = min(target_h, max(1, int(round(source_h * scale))))
        new_w = min(target_w, max(1, int(round(source_w * scale))))
        resized = resize_image(image, (new_h, new_w), interpolation, align_corners)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        return F.pad(resized, padding, mode="constant", value=float(fill_value))
    if normalized_mode in ("crop", "fill"):
        scale = max(float(target_h) / source_h, float(target_w) / source_w)
        new_h = max(target_h, int(round(source_h * scale)))
        new_w = max(target_w, int(round(source_w * scale)))
        resized = resize_image(image, (new_h, new_w), interpolation, align_corners)
        return center_crop(resized, (target_h, target_w))
    raise ValueError("aspect-ratio mode must be 'letterbox' or 'crop', got {!r}".format(mode))


def normalize(
    image: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    """Channel-wise normalize a floating CHW/BCHW tensor."""
    _validate_image(image)
    _require_float(image)
    channels = image.shape[-3]
    if len(mean) != channels or len(std) != channels:
        raise ValueError("mean and std must each have {} entries".format(channels))
    if any(float(value) <= 0.0 for value in std):
        raise ValueError("all standard deviations must be positive")
    view_shape = [1] * (image.ndim - 3) + [channels, 1, 1]
    mean_tensor = image.new_tensor(tuple(float(value) for value in mean)).view(view_shape)
    std_tensor = image.new_tensor(tuple(float(value) for value in std)).view(view_shape)
    return (image - mean_tensor) / std_tensor


def denormalize(
    image: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> Tensor:
    """Invert :func:`normalize`."""
    _validate_image(image)
    _require_float(image)
    channels = image.shape[-3]
    if len(mean) != channels or len(std) != channels:
        raise ValueError("mean and std must each have {} entries".format(channels))
    if any(float(value) <= 0.0 for value in std):
        raise ValueError("all standard deviations must be positive")
    view_shape = [1] * (image.ndim - 3) + [channels, 1, 1]
    mean_tensor = image.new_tensor(tuple(float(value) for value in mean)).view(view_shape)
    std_tensor = image.new_tensor(tuple(float(value) for value in std)).view(view_shape)
    return image * std_tensor + mean_tensor


def preprocess_image(
    image: Tensor,
    output_size: Optional[SizeLike] = None,
    center_crop_size: Optional[SizeLike] = None,
    keep_aspect_ratio: bool = False,
    aspect_mode: str = "letterbox",
    interpolation: str = "bilinear",
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
) -> Tensor:
    """Prepare an RGB tensor while keeping the project-wide RGB convention.

    UInt8 input is converted to float32 in ``[0,1]``.  Floating input is kept
    in its original dtype and is expected to already use ``[0,1]``.
    """
    _validate_image(image, channels=3)
    if image.dtype == torch.uint8:
        result = image.to(dtype=torch.float32) / 255.0
    elif image.is_floating_point():
        result = image
    else:
        raise TypeError("image must be uint8 or floating point")
    if center_crop_size is not None:
        result = center_crop(result, center_crop_size)
    if output_size is not None:
        if keep_aspect_ratio:
            result = resize_with_aspect_ratio(
                result,
                output_size,
                mode=aspect_mode,
                interpolation=interpolation,
            )
        else:
            result = resize_image(result, output_size, mode=interpolation, align_corners=False)
    if (normalize_mean is None) != (normalize_std is None):
        raise ValueError("normalize_mean and normalize_std must be provided together")
    if normalize_mean is not None and normalize_std is not None:
        result = normalize(result, normalize_mean, normalize_std)
    return result


def preprocess_pair(
    wide: Tensor,
    tele: Tensor,
    output_size: SizeLike,
    wide_crop_size: Optional[SizeLike] = None,
    keep_aspect_ratio: bool = False,
    aspect_mode: str = "letterbox",
    interpolation: str = "bilinear",
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
) -> Tuple[Tensor, Tensor]:
    """Preprocess synchronized Wide/Tele RGB tensors to one output size.

    The optional centre crop is applied only to Wide, matching the usual
    hybrid-zoom field-of-view setup.  Both returned tensors have the same rank
    and spatial dimensions.
    """
    _validate_image(wide, channels=3)
    _validate_image(tele, channels=3)
    if wide.ndim != tele.ndim:
        raise ValueError("wide and tele must both be CHW or both be BCHW")
    if wide.ndim == 4 and wide.shape[0] != tele.shape[0]:
        raise ValueError("wide and tele batch sizes must match")
    wide_result = preprocess_image(
        wide,
        output_size=output_size,
        center_crop_size=wide_crop_size,
        keep_aspect_ratio=keep_aspect_ratio,
        aspect_mode=aspect_mode,
        interpolation=interpolation,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    tele_result = preprocess_image(
        tele,
        output_size=output_size,
        keep_aspect_ratio=keep_aspect_ratio,
        aspect_mode=aspect_mode,
        interpolation=interpolation,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    return wide_result, tele_result


class ImagePreprocessor(nn.Module):
    """Configurable ``nn.Module`` wrapper around :func:`preprocess_pair`."""

    def __init__(
        self,
        output_size: SizeLike,
        wide_crop_size: Optional[SizeLike] = None,
        keep_aspect_ratio: bool = False,
        aspect_mode: str = "letterbox",
        interpolation: str = "bilinear",
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.output_size = _parse_size(output_size, "output_size")
        self.wide_crop_size = (
            None if wide_crop_size is None else _parse_size(wide_crop_size, "wide_crop_size")
        )
        self.keep_aspect_ratio = bool(keep_aspect_ratio)
        self.aspect_mode = aspect_mode
        self.interpolation = interpolation
        self.normalize_mean = None if normalize_mean is None else tuple(normalize_mean)
        self.normalize_std = None if normalize_std is None else tuple(normalize_std)

    def forward(self, wide: Tensor, tele: Tensor) -> Tuple[Tensor, Tensor]:
        return preprocess_pair(
            wide,
            tele,
            output_size=self.output_size,
            wide_crop_size=self.wide_crop_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
            aspect_mode=self.aspect_mode,
            interpolation=self.interpolation,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        )


# A concise compatibility alias used by a few research scripts.
resize_pair = preprocess_pair

