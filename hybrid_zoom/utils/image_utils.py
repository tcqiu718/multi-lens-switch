"""RGB image I/O with metric compatibility aliases."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .metrics import psnr, ssim

PathLike = Union[str, Path]


def pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert PIL RGB/L image to float32 CHW tensor in [0, 1]."""

    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def tensor_to_pil(image: Tensor) -> Image.Image:
    """Convert a CHW (or singleton BCHW) tensor to an RGB/L PIL image."""

    image = image.detach().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("tensor_to_pil accepts only a singleton batch")
        image = image[0]
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3 or image.shape[0] not in {1, 3, 4}:
        raise ValueError("Expected HW/CHW tensor with 1, 3, or 4 channels")
    array = (
        image.float().clamp(0.0, 1.0).mul(255.0).round().byte().permute(1, 2, 0).numpy()
    )
    if array.shape[2] == 1:
        return Image.fromarray(array[:, :, 0], mode="L")
    return Image.fromarray(array, mode="RGB" if array.shape[2] == 3 else "RGBA")


def read_image(
    path: PathLike,
    add_batch: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    """Read an image as RGB float32 CHW (or BCHW) in [0, 1]."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("Image does not exist: {}".format(source))
    with Image.open(source) as image:
        tensor = pil_to_tensor(image.convert("RGB"))
    if add_batch:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device=device) if device is not None else tensor


def save_image(image: Tensor, path: PathLike) -> Path:
    """Save an RGB or grayscale tensor, creating parent directories."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(destination)
    return destination


load_image = read_image
write_image = save_image
compute_psnr = psnr
compute_ssim = ssim

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "load_image",
    "pil_to_tensor",
    "psnr",
    "read_image",
    "save_image",
    "ssim",
    "tensor_to_pil",
    "write_image",
]
