"""RGB float32 image and mask I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def read_image(
    path: str,
    device: Optional[torch.device] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Read an image as RGB float32 [1,3,H,W] in [0,1]."""
    bgr = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("Could not read image: %s" % path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize is not None:
        height, width = int(resize[0]), int(resize[1])
        if height <= 0 or width <= 0:
            raise ValueError("resize must contain positive (height,width)")
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device=device) if device is not None else tensor


def read_mask(
    path: str,
    size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    gray = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError("Could not read mask: %s" % path)
    if size is not None and gray.shape != tuple(size):
        gray = cv2.resize(gray, (int(size[1]), int(size[0])), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
    return tensor.to(device=device) if device is not None else tensor


def tensor_to_rgb8(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Only one image can be converted at a time")
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("image must be [1/3,H,W] or [1,1/3,H,W]")
    array = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.round(array * 255.0).astype(np.uint8)


def write_image(path: str, image: torch.Tensor) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = tensor_to_rgb8(image)
    if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError("Could not write image: %s" % output)
    return output


def write_mask(path: str, mask: torch.Tensor, normalize: bool = False) -> Path:
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("mask must reduce to [H,W]")
    array = mask.detach().float().cpu().numpy()
    if normalize:
        minimum, maximum = float(array.min()), float(array.max())
        array = (array - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(array)
    array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), array):
        raise IOError("Could not write mask: %s" % output)
    return output


def validate_image_pair(wide: torch.Tensor, tele: torch.Tensor) -> None:
    if wide.shape != tele.shape:
        raise ValueError(
            "Wide/Tele resolution mismatch: %s versus %s. Calibrate/resize explicitly."
            % (tuple(wide.shape), tuple(tele.shape))
        )
    if wide.ndim != 4 or wide.shape[1] != 3:
        raise ValueError("Images must have shape [B,3,H,W]")
    if not torch.isfinite(wide).all() or not torch.isfinite(tele).all():
        raise ValueError("Input images contain NaN or Inf")


__all__ = ["read_image", "read_mask", "tensor_to_rgb8", "validate_image_pair", "write_image", "write_mask"]

